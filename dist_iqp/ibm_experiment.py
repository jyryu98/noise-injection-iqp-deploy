from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from qiskit import QuantumCircuit, ClassicalRegister
import networkx as nx
import numpy as np
import json
import pennylane as qml
import jax
import jax.numpy as jnp
from collections import defaultdict
from ast import literal_eval
from skopt import gp_minimize
from functools import partial
from .analysis import calculate_multi_mmd

class NoiseModeler:
    
    def __init__(self, num_training_circuits, num_val_circuits, shots_per_run, qpu_name):

        self.fragment_indices = [
            [0, 1, 2, 3, 16, 22, 23, 24],
            [4, 5, 6, 7, 17, 26, 27, 28],
            [8, 9, 10, 11, 18, 30, 31, 32],
            [12, 13, 14, 15, 19, 34, 35],
            [20, 21, 36, 40, 41, 42],
            [25, 37, 44, 45, 46],
            [29, 38, 48, 49, 50],
            [33, 39, 52, 53, 54],
            [43, 56, 62, 63, 64],
            [47, 57, 66, 67, 68],
            [51, 58, 70, 71, 72],
            [55, 59, 74, 75],
            [60, 61, 76, 80, 81, 82],
            [65, 77, 84, 85, 86],
            [69, 78, 88, 89, 90],
            [73, 79, 92, 93, 94],
            [83, 96, 102, 103, 104],
            [87, 97, 106, 107, 108],
            [91, 98, 110, 111, 112],
            [95, 99, 114, 115],
            [100, 101, 116, 120, 121, 122],
            [105, 117, 124, 125, 126],
            [109, 118, 128, 129, 130],
            [113, 119, 132, 133, 134],
            [123, 136, 140, 141, 142, 143, 144],
            [127, 137, 145, 146, 147, 148],
            [131, 138, 149, 150, 151, 152],
            [135, 139, 153, 154, 155]
        ]

        self.flip_lattice_map = {i: 140+i for i in range(16)} | {i: 120+i for i in range(16, 20)} | {i: 100 + i for i in range(20, 36)} | {i: 80 + i for i in range(36, 40)} | {i: 60 + i for i in range(40, 56)} | {i: 40 + i for i in range(56, 60)} | {i: 20 + i for i in range(60, 76)} | {i: i for i in range(76, 80)}
        self.flip_lattice_map = self.flip_lattice_map |  {self.flip_lattice_map[i]: i for i in self.flip_lattice_map.keys()}
        self.flip_fragment_indices = [[self.flip_lattice_map[l] for l in k] for k in self.fragment_indices]

        self.num_training_circuits = num_training_circuits 
        self.num_val_circuits = num_val_circuits
        self.shots_per_run = shots_per_run
        self.qpu_name = qpu_name

    def noise_modeling_qpu_runs(self,):

        # Get connectivity
        backend = FakeMarrakesh()
        physical_edges = []
        for e in backend.coupling_map:
            if e not in physical_edges and e[::-1] not in physical_edges:
                physical_edges.append(e)

        # Generate layer map
        interaction_graph = nx.Graph()
        interaction_graph.add_nodes_from(range(backend.num_qubits))
        for physical_edge in physical_edges:
            interaction_graph.add_edge(physical_edge[0], physical_edge[1])

        lineG = nx.line_graph(interaction_graph)
        layer_map = nx.greedy_color(lineG, strategy = "smallest_last")
        num_layers = max([layer_map[i] for i in layer_map.keys()])+1
        if num_layers != 3:
            raise ValueError
        with open('layer_map.json', 'w') as f:
            json.dump({str(k): layer_map[k] for k in layer_map.keys()}, f)

        # Get service details
        service = QiskitRuntimeService(token = 'TOKEN')
        backend_fractional = service.backend(self.qpu_name, use_fractional_gates=True)
        initial_qubit_map = list(range(backend.num_qubits))
        fragment_pm = generate_preset_pass_manager(
            optimization_level=3, backend=backend_fractional, scheduling_method="alap", initial_layout = initial_qubit_map
        )

        # Construct, save, and transpile circuits 
        circuits_to_run = []
        for runidx in range(self.num_training_circuits + self.num_val_circuits):
            for config in ['reg', 'flip']:

                # Sample rotation angles
                rot_angles = {}
                for qidx in range(backend.num_qubits):
                    rot_angles[str(qidx)] = 2*np.pi*np.random.rand()
                for physical_edge in layer_map.keys():
                    if config == 'reg':
                        for fragment in self.fragment_indices:
                            if physical_edge[0] in fragment and physical_edge[1] in fragment:
                                rot_angles[str(physical_edge)] = 0.5*np.pi*np.random.rand()
                    elif config == 'flip':
                        for fragment in self.flip_fragment_indices:
                            if physical_edge[0] in fragment and physical_edge[1] in fragment:
                                rot_angles[str(physical_edge)] = 0.5*np.pi*np.random.rand()
                    else: raise ValueError

                with open('run_{}_config_{}_rot_angles.json'.format(runidx, config), 'w') as f:
                    json.dump(rot_angles, f)

                # Make quantum circuit
                qc = QuantumCircuit(backend.num_qubits)
                for i in range(backend.num_qubits):
                    qc.h(i)
                    qc.rz(rot_angles[str(i)], i)
                qc.barrier()
                for l in range(num_layers):
                    for e in layer_map.keys():
                        if str(e) in list(rot_angles.keys()) and layer_map[e] == l:
                            qc.rzz(rot_angles[str(e)], e[0], e[1])
                    qc.barrier()
                for i in range(backend.num_qubits):
                    qc.h(i)
                qc.measure_all()

                # Transpile
                transpiled_qc = fragment_pm.run(qc)
                circuits_to_run.append(transpiled_qc)

        # Run
        sampler = Sampler(mode=backend_fractional)
        job = sampler.run(circuits_to_run, shots = self.shots_per_run)
        result = job.result()

        # Save
        result_idx = 0
        for runidx in range(self.num_training_circuits + self.num_val_circuits):
            for config in ['reg', 'flip']:
                with open('run_{}_config_{}_bitstrings.json'.format(runidx, config), 'w') as file:
                    json.dump(result[result_idx].data.meas.get_counts(), file)
                result_idx += 1

    def single_qubit_Pauli_channel(self, noise_parameters, wire):
        """Apply a single qubit Pauli channel
        
        Args:
            noise_parameters (np.array): 1D, length 3 array of noise parameters (px, py, pz)
            wire (int): Qubit index to place the channel on
        """
        kraus_ops = [jnp.sqrt(1-jnp.sum(noise_parameters))*jnp.eye(2),
                    jnp.sqrt(noise_parameters[0])*qml.PauliX.compute_matrix(),
                    jnp.sqrt(noise_parameters[1])*qml.PauliY.compute_matrix(),
                    jnp.sqrt(noise_parameters[2])*qml.PauliZ.compute_matrix()]
        qml.QubitChannel(kraus_ops, wires = [wire])

    def construct_circuit(self, rot_angles, layer_map, fragment_indices):
        num_qubits = len(fragment_indices)
        dev = qml.device("default.mixed", wires = num_qubits)
        
        gate_layer_angle_info = []
        for l in range(self.num_training_circuits):
            layer = []
            for gate in layer_map.keys():
                gate_tuple = literal_eval(gate)
                if layer_map[gate] == l and gate_tuple[0] in fragment_indices and gate_tuple[1] in fragment_indices:
                    layer.append([[fragment_indices.index(gate_tuple[0]), fragment_indices.index(gate_tuple[1])], rot_angles[gate]])
            gate_layer_angle_info.append(layer)

        @qml.qnode(dev, interface='jax')
        def fragment_noisy_model(p_noise):
            for i in range(num_qubits):
                qml.Hadamard(wires = i)
                qml.RZ(rot_angles[str(fragment_indices[i])], wires = i)
                self.single_qubit_Pauli_channel(p_noise, i)
            for layer in gate_layer_angle_info:
                for gate, angle in layer:
                    qml.IsingZZ(angle, wires = gate)
                for i in range(num_qubits):
                    self.single_qubit_Pauli_channel(p_noise, i)
            for i in range(num_qubits):
                qml.Hadamard(wires = i)
            return qml.probs(wires = range(num_qubits))
        
        jitted_circuit = jax.jit(fragment_noisy_model)
        return jitted_circuit

    def bitstrings_to_fragment_epdfs(self, bitstrings, fragment_indices):
        epdfs = []
        for frag in fragment_indices:
            marginal = defaultdict(int)
            for bitstring, count in bitstrings.items():
                reverse_bitstring = bitstring[::-1]
                projected = "".join(reverse_bitstring[i] for i in frag)
                marginal[projected] += count
            epdf = np.zeros(2**len(frag))
            total_shots = np.sum(np.array([marginal[k] for k in marginal.keys()]))
            for bitstring in marginal.keys():
                epdf[int(bitstring, 2)] += marginal[bitstring]/total_shots
            epdfs.append(epdf)
        return epdfs 

    def fit_noise_model(self, depol_only = False):
        # Load layer map
        with open('layer_map.json', 'r') as f:
            layer_map = json.load(f)

        # Preprocessing
        jitted_circuits = []
        epdfs = []
        for runidx in range(self.num_training_circuits):
            for configuration in ['reg', 'flip']:
                # 1. JIT compilation of fragments
                with open('run_{}_config_{}_rot_angles.json'.format(runidx, configuration), 'r') as f:
                    rot_angles = json.load(f)
                if configuration == 'reg':
                    for frag in self.fragment_indices:
                        jitted_circuits.append(self.construct_circuit(rot_angles, layer_map, frag))
                elif configuration == 'flip':
                    for frag in self.flip_fragment_indices:
                        jitted_circuits.append(self.construct_circuit(rot_angles, layer_map, frag))

                # 2. Bitstring, counts -> marginal empirical PDFs
                with open('./run_{}_config_{}_bitstrings.json'.format(runidx, configuration), 'r') as f:
                    bitstrings = json.load(f)
                if configuration == 'reg':
                    epdfs = epdfs + self.bitstrings_to_fragment_epdfs(bitstrings, self.fragment_indices)
                elif configuration == 'flip':
                    epdfs = epdfs + self.bitstrings_to_fragment_epdfs(bitstrings, self.flip_fragment_indices)
        
        # Validation Preprocessing
        val_jitted_circuits = []
        val_epdfs = []
        for runidx in range(self.num_training_circuits, self.num_training_circuits + self.num_val_circuits):
            for configuration in ['reg', 'flip']:
                # 1. JIT compilation of fragments
                with open('run_{}_config_{}_rot_angles.json'.format(runidx, configuration), 'r') as f:
                    rot_angles = json.load(f)
                if configuration == 'reg':
                    for frag in self.fragment_indices:
                        val_jitted_circuits.append(self.construct_circuit(rot_angles, layer_map, frag))
                elif configuration == 'flip':
                    for frag in self.flip_fragment_indices:
                        val_jitted_circuits.append(self.construct_circuit(rot_angles, layer_map, frag))

                # 2. Bitstring, counts -> marginal empirical PDFs
                with open('./run_{}_config_{}_bitstrings.json'.format(runidx, configuration), 'r') as f:
                    bitstrings = json.load(f)
                if configuration == 'reg':
                    val_epdfs = val_epdfs + self.bitstrings_to_fragment_epdfs(bitstrings, self.fragment_indices)
                elif configuration == 'flip':
                    val_epdfs = val_epdfs + self.bitstrings_to_fragment_epdfs(bitstrings, self.flip_fragment_indices)
        print('JIT compiling circuits and marginal empirical pdf calculation complete')
        print('Fitting noise models..\n')

        ######################################################
        # Case 1: Depolarizing model (1 parameter)
        ######################################################

        def depol_noise_fitting_objective(p_noise):
            mmds = []
            for jc, epdf in zip(jitted_circuits, epdfs):
                model_pdf = jc(np.array([p_noise[0], p_noise[0], p_noise[0]]))
                num_qubits = int(np.log2(len(epdf)))
                mmd = calculate_multi_mmd(model_pdf, epdf, num_qubits)
                mmds.append(mmd)
            return np.mean(mmds)

        def depol_val_noise_fitting_objective(p_noise):
            mmds = []
            for jc, epdf in zip(val_jitted_circuits, val_epdfs):
                model_pdf = jc(np.array([p_noise[0], p_noise[0], p_noise[0]]))
                num_qubits = int(np.log2(len(epdf)))
                mmd = calculate_multi_mmd(model_pdf, epdf, num_qubits)
                mmds.append(mmd)
            return np.mean(mmds)
        
        # Grid search
        ps = np.linspace(0, 0.05, 100)
        objs = [depol_noise_fitting_objective([p]) for p in ps]

        minimum_p = ps[objs.index(min(objs))]
        minimum_train_obj = depol_noise_fitting_objective([minimum_p])
        val_obj = depol_val_noise_fitting_objective([minimum_p])

        np.save('depol_min_p.npy', np.array(minimum_p))
        np.save('depol_min_train_obj.npy', np.array(minimum_train_obj))
        np.save('depol_val_obj.npy', np.array(val_obj))
        np.save('depol_grid_objs.npy', np.array(objs))

        print('Depolarizing Noise Model Fit Results:')
        print('Argmin p: {}\nMinimum train obj: {}\nVal obj: {}\n'.format(minimum_p, minimum_train_obj, val_obj))

        if not depol_only:
            ######################################################
            # Case 2: Bitflip+Dephasing model (2 parameters)
            ######################################################

            def bitflip_dephasing_noise_fitting_objective(p_noise):
                mmds = []
                for jc, epdf in zip(jitted_circuits, epdfs):
                    model_pdf = jc(np.array([p_noise[0]*(1-p_noise[1]), p_noise[0]*p_noise[1], p_noise[1]*(1-p_noise[0])]))
                    num_qubits = int(np.log2(len(epdf)))
                    mmd = calculate_multi_mmd(model_pdf, epdf, num_qubits)
                    mmds.append(mmd)
                return np.mean(mmds)

            def bitflip_dephasing_val_noise_fitting_objective(p_noise):
                mmds = []
                for jc, epdf in zip(val_jitted_circuits, val_epdfs):
                    model_pdf = jc(np.array([p_noise[0]*(1-p_noise[1]), p_noise[0]*p_noise[1], p_noise[1]*(1-p_noise[0])]))
                    num_qubits = int(np.log2(len(epdf)))
                    mmd = calculate_multi_mmd(model_pdf, epdf, num_qubits)
                    mmds.append(mmd)
                return np.mean(mmds)
            
            # Grid search
            pxs = np.linspace(0, 0.05, 50)
            pzs = np.linspace(0, 0.05, 50)
            objs = np.zeros((50, 50))
            for pxi, px in enumerate(pxs):
                for pzi, pz in enumerate(pzs):
                    objs[pxi, pzi] = bitflip_dephasing_noise_fitting_objective([px, pz])

            min_pxi, min_pzi = np.unravel_index(objs.argmin(), objs.shape)
            min_p = [pxs[min_pxi], pzs[min_pzi]]
            minimum_train_obj = bitflip_dephasing_noise_fitting_objective(min_p)
            val_obj = bitflip_dephasing_val_noise_fitting_objective(min_p)

            np.save('nxnz_min_p.npy', np.array(min_p))
            np.save('nxnz_min_train_obj.npy', np.array(minimum_train_obj))
            np.save('nxnz_val_obj.npy', np.array(val_obj))
            np.save('nxnz_grid_objs.npy', np.array(objs))

            print('Bitflip+Dephasing Noise Model Fit Results:')
            print('Argmin p: {}\nMinimum train obj: {}\nVal obj: {}\n'.format(min_p, minimum_train_obj, val_obj))

            ######################################################
            # Case 3: Aribtrary 1Q Pauli model (3 parameters)
            ######################################################

            def pauli_noise_fitting_objective(p_noise):
                mmds = []
                for jc, epdf in zip(jitted_circuits, epdfs):
                    model_pdf = jc(np.array(p_noise))
                    num_qubits = int(np.log2(len(epdf)))
                    mmd = calculate_multi_mmd(model_pdf, epdf, num_qubits)
                    mmds.append(mmd)
                return np.mean(mmds)

            def pauli_val_noise_fitting_objective(p_noise):
                mmds = []
                for jc, epdf in zip(val_jitted_circuits, val_epdfs):
                    model_pdf = jc(np.array(p_noise))
                    num_qubits = int(np.log2(len(epdf)))
                    mmd = calculate_multi_mmd(model_pdf, epdf, num_qubits)
                    mmds.append(mmd)
                return np.mean(mmds)
            
            # Bayesian optimization
            res = gp_minimize(pauli_noise_fitting_objective,    # the function to minimize
                        [(0, 0.1), (0, 0.1), (0, 0.1)], # the bounds on each dimension of x
                        acq_func="EI",      # the acquisition function
                        n_calls=50,         # the number of evaluations of f
                        n_initial_points=5,  # the number of random initialization points
                        noise=0.1**2,       # the noise level (optional)
                        random_state=1234,  # the random seed
                        verbose=False)

            min_p = res.x
            minimum_train_obj = pauli_noise_fitting_objective(min_p)
            val_obj = pauli_val_noise_fitting_objective(min_p)

            np.save('sqpc_min_p.npy', np.array(min_p))
            np.save('sqpc_min_train_obj.npy', np.array(minimum_train_obj))
            np.save('sqpc_val_obj.npy', np.array(val_obj))
            np.save('sqpc_bo_objs.npy', np.array(res.func_vals))
            np.save('sqpc_bo_x_iters.npy', np.array(res.x_iters))

            print('Arbitrary 1Q Pauli Channel Noise Model Fit Results:')
            print('Argmin p: {}\nMinimum train obj: {}\nVal obj: {}'.format(min_p, minimum_train_obj, val_obj))

        return 
    
class IBMPercolationDistributor:

    def __init__(self, G, noise_parameters, max_classical_sim_size, qpu_name):
        """Simulate a noisy IQP circuit using smaller, less noisy quantum computers.

        Args:
            G (nx.Graph): Interaction graph of the circuit. Requires node and edge attributes of name 'weight',
                which are the rotation angles of the corresponding gates.
            noise_parameters (np.array): Noise paramters [p_x, p_y, p_z] for the layer-wise Pauli channel acting on all qubits for distributed nodes.
                $N(\rho) = (1-p_x-p_y-p_z)\rho + p_x X\rho X + p_y Y\rho Y + p_z Z\rho Z$.
                Note that the values should add up to a value less than 1.
            
        """

        self.max_classical_sim_size = max_classical_sim_size
        self.qpu_name = qpu_name
        self.interaction_graph = G
        self.num_qubits = len(G)
        self.q1_weights = nx.get_node_attributes(G, 'weight')
        self.q2_weights = nx.get_edge_attributes(G, 'weight')

        if len(noise_parameters.shape) > 1 or len(noise_parameters) != 3 or sum(noise_parameters) > 1:
            raise ValueError
        self.noise_parameters = noise_parameters

        # Divide interaction graph into layers via edge coloring
        lineG = nx.line_graph(G)
        self.layer_map = nx.greedy_color(lineG)
        self.num_layers = max([self.layer_map[i] for i in self.layer_map.keys()])+2

        # Get service details
        self.service = QiskitRuntimeService(token = 'TOKEN')
        self.backend_fractional = self.service.backend(qpu_name, use_fractional_gates=True)
        #self.initial_qubit_map = list(range(self.backend_fractional.num_qubits))
        self.fragment_pm = generate_preset_pass_manager(
            optimization_level=3, backend=self.backend_fractional, scheduling_method="alap"
        )
        self.max_circuits_per_job = self.backend_fractional.configuration().max_experiments

    def _single_qubit_Pauli_channel(self, noise_parameters, wire):
        """Apply a single qubit Pauli channel
        
        Args:
            noise_parameters (np.array): 1D, length 3 array of noise parameters (px, py, pz)
            wire (int): Qubit index to place the channel on
        """
        kraus_ops = [np.sqrt(1-np.sum(noise_parameters))*np.eye(2),
                     np.sqrt(noise_parameters[0])*qml.PauliX.compute_matrix(),
                     np.sqrt(noise_parameters[1])*qml.PauliY.compute_matrix(),
                     np.sqrt(noise_parameters[2])*qml.PauliZ.compute_matrix()]
        qml.QubitChannel(kraus_ops, wires = [wire])

    def _percolation(self, percolation_p):
        """
        Randomly choose nodes to occupy given bitflip noise probability and output connected components

        Args:
            percolation_p (float): Bitflip noise probability.

        Returns:
            partition (list): List of sets of qubit numbers indicating connected clusters
        """

        # Choose where completely dephasing noise occurs
        survival_p = 1-percolation_p
        dephasing_noise_filter = np.random.binomial(1, survival_p, size = (self.num_qubits))

        # Partition circuit
        sample_graph = self.interaction_graph.copy()
        for qubit_idx, dephasing_noise in enumerate(dephasing_noise_filter):
            if dephasing_noise == 0:
                sample_graph.remove_node(qubit_idx)
        connectedGs = nx.connected_components(sample_graph)
        connectedGs = [g for g in connectedGs]

        if len(connectedGs) == 0:
            # If there are no connected components, all qubits are separated
            partition = [{i} for i in range(self.num_qubits)]
        else:
            # Add 1 qubit segments to the 2+ qubit segments
            partition = []
            connected_qubits = set()
            for subcircuit in connectedGs:
                connected_qubits = connected_qubits.union(subcircuit)
            for i in range(self.num_qubits):
                if i not in connected_qubits:
                    partition.append({i})
            partition = partition + connectedGs

        return partition
    
    def _fine_tune_percolation_p(self, initial_percolation_p, capacity, n_reps):
        p = initial_percolation_p
        threshold_fraction = 0.9
        increase_step = 0.001
        meet_capacity = False 

        while not meet_capacity:
            num_capacity_samples = 0
            for _ in range(n_reps):
                partition = self._percolation(p)
                if max([len(s) for s in partition]) <= capacity:
                    num_capacity_samples += 1
            meet_capacity = num_capacity_samples >= threshold_fraction * n_reps
            p += increase_step 
        return p-increase_step

    def effective_noise_parameters(self, percolation_p):
        """Calculate effective noise parameters of the simulated large quantum computer
        
        Args:
            percolation_p (float):
        """

        layerwise_percolation_p = 1-(1-percolation_p)**(1/self.num_layers)
        effective_noise_parameters = [(1-layerwise_percolation_p)*self.noise_parameters[0],
                                      (1-layerwise_percolation_p)*self.noise_parameters[1],
                                      (1-layerwise_percolation_p)*self.noise_parameters[2]+0.5*layerwise_percolation_p]
        return effective_noise_parameters
    
    def _concatenate_sample(self, ns, mode):
        """Concatenates results from self._sample_noisy_circuit into a decimal number

        Args:
            ns (dict[str:int|list[int]]): Sampled bits for corresponding qubits
            mode (str): 'decimal' or 'binary'
        Returns:
            (int): Concatenated result as a decimal number
        """
        sample = []
        for i in range(self.num_qubits):
            for k in ns.keys():
                if str(i) in k.split('|') and len(k.split('|')) == 1:
                    if np.isscalar(ns[k]):
                        sample.append(ns[k]) # Removed node
                    else:
                        sample.append(int(ns[k].flatten()[0])) # Cluster is size 1
                elif str(i) in k.split('|') and len(k.split('|')) > 1:
                    sample.append(int(np.squeeze(ns[k])[k.split('|').index(str(i))]))
        if mode == 'binary':
            return ''.join([str(i) for i in sample])
        if mode == 'decimal':
            return int(''.join([str(i) for i in sample]), 2)
    
    def sample(self, num_samples, percolation_p, max_capacity):
        
        layerwise_percolation_p = 1-(1-percolation_p)**(1/self.num_layers)
        
        circuits_to_run_qpu = []
        noisy_circuit_samples = []
        bitflip_locations_sample = []

        num_ooc_shots = 0
        for sample_idx in range(num_samples):

            # Sample noise matrix and identify removed nodes
            noise_matrix = np.random.binomial(1, p = layerwise_percolation_p, size = (self.num_qubits, self.num_layers))
            removed_nodes = np.argwhere(np.sum(noise_matrix, axis = 1) > 0).flatten()
            bitflip_locations = (np.sum(noise_matrix, axis = 1) > 0).astype(int)
            noisy_circuit_sample = {str(n): np.random.binomial(1, p = 0.5) for n in removed_nodes}

            # Partition circuit
            sample_graph = self.interaction_graph.copy()
            sample_graph.remove_nodes_from(removed_nodes)
            connectedGs = nx.connected_components(sample_graph)
            connectedGs = [g for g in connectedGs]

            if len(connectedGs) == 0:
                # If there are no connected components, all qubits are separated
                partition = [{i} for i in range(self.num_qubits)]
            else:
                # Add 1 qubit segments to the 2+ qubit segments
                partition = []
                connected_qubits = set()
                for subcircuit in connectedGs:
                    connected_qubits = connected_qubits.union(subcircuit)
                for i in range(self.num_qubits):
                    if i not in connected_qubits:
                        partition.append({i})
                partition = partition + connectedGs

            # Check capacity
            if max([len(s) for s in partition]) <= max_capacity:
                # Simulate single qubit noise
                gate_angle_multiplier = {}
                for layer_idx in range(self.num_layers):
                    if layer_idx > 0: # Exclude RZ layer
                    # Calculate removed node impact on gates
                    # If both qubits are removed, multiplier = 0
                    # If one qubit is removed, multiplier = (-1)^b
                        for e in self.interaction_graph.edges():
                            if layer_idx-1 == self.layer_map[e]: # Exclude RZ layer
                                if e[0] in removed_nodes:
                                    if e[1] in removed_nodes:
                                        gate_angle_multiplier[e] = 0
                                    else: gate_angle_multiplier[e] = (-1)**noisy_circuit_sample[str(e[0])]
                                else:
                                    if e[1] in removed_nodes:
                                        gate_angle_multiplier[e] = (-1)**noisy_circuit_sample[str(e[1])]
                                    else: gate_angle_multiplier[e] = 1

                    # Update random bit by simulating 1Q noise channel
                    for qubit_idx in removed_nodes:
                        if noise_matrix[qubit_idx, layer_idx] == 0:
                            noisy_circuit_sample[str(qubit_idx)] = (noisy_circuit_sample[str(qubit_idx)]
                                                                    + np.random.binomial(1, p = self.noise_parameters[0]+self.noise_parameters[1]))%2
                # Randomize removed nodes
                for qubit_idx in removed_nodes:
                    noisy_circuit_sample[str(qubit_idx)] = np.random.binomial(1, p = 0.5)
                
                # Simulate each cluster
                for p in partition:
                    # 1. If fragment size is small, simulate classically
                    if len(p) <= self.max_classical_sim_size:
                        if len(p) > 1 or (len(p) == 1 and list(p)[0] not in removed_nodes):
                            dev = qml.device('default.mixed', wires = list(p))
                            @partial(qml.set_shots, shots=1)
                            @qml.qnode(device = dev)
                            def partition_circuit():
                                # Single qubit gates
                                for i in list(p):
                                    qml.Hadamard(i)
                                    qml.RZ(self.q1_weights[i], i)
                                    self._single_qubit_Pauli_channel(self.noise_parameters, i)
                                # Two qubit gates
                                for layernum in range(self.num_layers-1):
                                    for e in self.interaction_graph.edges():
                                        if self.layer_map[e] == layernum:
                                            if (e[0] in p) and (e[1] in p):
                                                qml.IsingZZ(self.q2_weights[e], [e[0], e[1]])
                                            elif (e[0] in p) and (e[1] not in p):
                                                qml.RZ(self.q2_weights[e] * gate_angle_multiplier[e], e[0])
                                            elif (e[1] in p) and (e[0] not in p):
                                                qml.RZ(self.q2_weights[e] * gate_angle_multiplier[e], e[1])
                                    for i in list(p):
                                        self._single_qubit_Pauli_channel(self.noise_parameters, i)
                                # Hadamard layer
                                for i in list(p):
                                    qml.Hadamard(i)
                                return qml.sample(wires = list(p))
                            noisy_circuit_sample['|'.join([str(i) for i in list(p)])] = partition_circuit()
                    # 2. Otherwise, use QPU
                    else:
                        # Build QC for QPU run
                        qc = QuantumCircuit(len(p))
                        for i in range(len(p)):
                            qc.h(i)
                            qc.rz(self.q1_weights[list(p)[i]], i)
                        qc.barrier()
                        for l in range(self.num_layers-1):
                            for e in self.interaction_graph.edges():
                                if self.layer_map[e] == l:
                                    if (e[0] in p) and (e[1] in p):
                                        qc.rzz(self.q2_weights[e], list(p).index(e[0]), list(p).index(e[1]))
                                    elif (e[0] in p) and (e[1] not in p):
                                        qc.rz(self.q2_weights[e] * gate_angle_multiplier[e], list(p).index(e[0]))
                                    elif (e[1] in p) and (e[0] not in p):
                                        qc.rz(self.q2_weights[e] * gate_angle_multiplier[e], list(p).index(e[1]))
                            qc.barrier()
                        for i in range(len(p)):
                            qc.h(i)
                        qc.measure_all()
                        circuits_to_run_qpu.append([{'sample_idx': sample_idx, 'partition': p}, qc])
                noisy_circuit_samples.append([{'sample_idx': sample_idx}, noisy_circuit_sample])
                bitflip_locations_sample.append([{'sample_idx': sample_idx}, bitflip_locations])
            else:
                num_ooc_shots += 1

        print("Out of {} Shots, {} Shots are out of capacity".format(num_samples, num_ooc_shots))
        print("Percolation created {} fragments larger than {} qubits".format(len(circuits_to_run_qpu), self.max_classical_sim_size))

        # Run by chunks of self.max_circuits_per_job
        chunk_circuits_to_run_qpu = [circuits_to_run_qpu[i:i + self.max_circuits_per_job] for i in range(0, len(circuits_to_run_qpu), self.max_circuits_per_job)]
        for chunk in chunk_circuits_to_run_qpu:
            
            # Transpile
            transpiled_qcs = [self.fragment_pm.run(k[1]) for k in chunk]

            # Run
            sampler = Sampler(mode=self.backend_fractional)
            job = sampler.run(transpiled_qcs, shots = 1)
            result = job.result()

            # Append QPU results into the appropriate noisy circuit sample
            for result_idx in range(len(transpiled_qcs)):
                bs = list(result[result_idx].data.meas.get_counts().keys())[0]
                bs = bs[::-1] # Due to changes in convention from LSB -> MSB
                p = chunk[result_idx][0]['partition']
                result_sample_idx = chunk[result_idx][0]['sample_idx']
                
                # Find corresponding noisy_circuit_sample
                for ncs_idx, ncs in enumerate(noisy_circuit_samples):
                    if ncs[0]['sample_idx'] == result_sample_idx:
                        result_ncs_idx = ncs_idx

                # Insert results into noisy_circuit_sample
                noisy_circuit_samples[result_ncs_idx]['|'.join([str(i) for i in list(p)])] = np.array([[int(b) for b in bs]])

        # Concat
        concatenated_samples = [self._concatenate_sample(s[1], 'binary') for s in noisy_circuit_samples]

        return concatenated_samples, [bl[1] for bl in bitflip_locations]

    def parallel_sample(self, num_samples, percolation_p, max_capacity):
            
        layerwise_percolation_p = 1-(1-percolation_p)**(1/self.num_layers)
        
        circuits_to_run_qpu = []
        noisy_circuit_samples = []
        bitflip_locations_sample = []

        num_ooc_shots = 0
        for sample_idx in range(num_samples):

            # Sample noise matrix and identify removed nodes
            noise_matrix = np.random.binomial(1, p = layerwise_percolation_p, size = (self.num_qubits, self.num_layers))
            removed_nodes = np.argwhere(np.sum(noise_matrix, axis = 1) > 0).flatten()
            bitflip_locations = (np.sum(noise_matrix, axis = 1) > 0).astype(int)
            noisy_circuit_sample = {str(n): np.random.binomial(1, p = 0.5) for n in removed_nodes}

            # Partition circuit
            sample_graph = self.interaction_graph.copy()
            sample_graph.remove_nodes_from(removed_nodes)
            connectedGs = nx.connected_components(sample_graph)
            connectedGs = [g for g in connectedGs]

            if len(connectedGs) == 0:
                # If there are no connected components, all qubits are separated
                partition = [{i} for i in range(self.num_qubits)]
            else:
                # Add 1 qubit segments to the 2+ qubit segments
                partition = []
                connected_qubits = set()
                for subcircuit in connectedGs:
                    connected_qubits = connected_qubits.union(subcircuit)
                for i in range(self.num_qubits):
                    if i not in connected_qubits:
                        partition.append({i})
                partition = partition + connectedGs

            # Check capacity
            if max([len(s) for s in partition]) <= max_capacity:
                # Simulate single qubit noise
                gate_angle_multiplier = {}
                for layer_idx in range(self.num_layers):
                    if layer_idx > 0: # Exclude RZ layer
                    # Calculate removed node impact on gates
                    # If both qubits are removed, multiplier = 0
                    # If one qubit is removed, multiplier = (-1)^b
                        for e in self.interaction_graph.edges():
                            if layer_idx-1 == self.layer_map[e]: # Exclude RZ layer
                                if e[0] in removed_nodes:
                                    if e[1] in removed_nodes:
                                        gate_angle_multiplier[e] = 0
                                    else: gate_angle_multiplier[e] = (-1)**noisy_circuit_sample[str(e[0])]
                                else:
                                    if e[1] in removed_nodes:
                                        gate_angle_multiplier[e] = (-1)**noisy_circuit_sample[str(e[1])]
                                    else: gate_angle_multiplier[e] = 1

                    # Update random bit by simulating 1Q noise channel
                    for qubit_idx in removed_nodes:
                        if noise_matrix[qubit_idx, layer_idx] == 0:
                            noisy_circuit_sample[str(qubit_idx)] = (noisy_circuit_sample[str(qubit_idx)]
                                                                    + np.random.binomial(1, p = self.noise_parameters[0]+self.noise_parameters[1]))%2
                # Randomize removed nodes
                for qubit_idx in removed_nodes:
                    noisy_circuit_sample[str(qubit_idx)] = np.random.binomial(1, p = 0.5)
                
                # Simulate each cluster
                for p in partition:
                    # 1. If fragment size is small, simulate classically
                    if len(p) <= self.max_classical_sim_size:
                        if len(p) > 1 or (len(p) == 1 and list(p)[0] not in removed_nodes):
                            dev = qml.device('default.mixed', wires = list(p))
                            @partial(qml.set_shots, shots=1)
                            @qml.qnode(device = dev)
                            def partition_circuit():
                                # Single qubit gates
                                for i in list(p):
                                    qml.Hadamard(i)
                                    qml.RZ(self.q1_weights[i], i)
                                    self._single_qubit_Pauli_channel(self.noise_parameters, i)
                                # Two qubit gates
                                for layernum in range(self.num_layers-1):
                                    for e in self.interaction_graph.edges():
                                        if self.layer_map[e] == layernum:
                                            if (e[0] in p) and (e[1] in p):
                                                qml.IsingZZ(self.q2_weights[e], [e[0], e[1]])
                                            elif (e[0] in p) and (e[1] not in p):
                                                qml.RZ(self.q2_weights[e] * gate_angle_multiplier[e], e[0])
                                            elif (e[1] in p) and (e[0] not in p):
                                                qml.RZ(self.q2_weights[e] * gate_angle_multiplier[e], e[1])
                                    for i in list(p):
                                        self._single_qubit_Pauli_channel(self.noise_parameters, i)
                                # Hadamard layer
                                for i in list(p):
                                    qml.Hadamard(i)
                                return qml.sample(wires = list(p))
                            noisy_circuit_sample['|'.join([str(i) for i in list(p)])] = partition_circuit()
                    # 2. Otherwise, use QPU
                    else:
                        # Build QC for QPU run
                        qc = QuantumCircuit(len(p))
                        for i in range(len(p)):
                            qc.h(i)
                            qc.rz(self.q1_weights[list(p)[i]], i)
                        qc.barrier()
                        for l in range(self.num_layers-1):
                            for e in self.interaction_graph.edges():
                                if self.layer_map[e] == l:
                                    if (e[0] in p) and (e[1] in p):
                                        qc.rzz(self.q2_weights[e], list(p).index(e[0]), list(p).index(e[1]))
                                    elif (e[0] in p) and (e[1] not in p):
                                        qc.rz(self.q2_weights[e] * gate_angle_multiplier[e], list(p).index(e[0]))
                                    elif (e[1] in p) and (e[0] not in p):
                                        qc.rz(self.q2_weights[e] * gate_angle_multiplier[e], list(p).index(e[1]))
                            qc.barrier()
                        for i in range(len(p)):
                            qc.h(i)
                        qc.measure_all()
                        circuits_to_run_qpu.append([{'sample_idx': sample_idx, 'partition': p}, qc])
                noisy_circuit_samples.append([{'sample_idx': sample_idx}, noisy_circuit_sample])
                bitflip_locations_sample.append([{'sample_idx': sample_idx}, bitflip_locations])
            else:
                num_ooc_shots += 1

        print("Out of {} Shots, {} Shots are out of capacity".format(num_samples, num_ooc_shots))
        print("Percolation created {} fragments larger than {} qubits".format(len(circuits_to_run_qpu), self.max_classical_sim_size))

        # If circuit size is between max_classical_sim_size and smaller than num_qubits/4, group into pairs to run in parallel
        pairable_circuits = []
        single_circuits_to_run_qpu = []
        for qc_with_info in circuits_to_run_qpu:
            if qc_with_info[1].num_qubits <= self.num_qubits/4:
                pairable_circuits.append(qc_with_info)
            else:
                single_circuits_to_run_qpu.append(qc_with_info)

        # Pair medium circuits conservatively.
        # Sort by size and pair largest with smallest.
        pairable_circuits.sort(key=lambda x: x[1].num_qubits)

        pairs = []
        left = 0
        right = len(pairable_circuits) - 1

        while left < right:
            # Since each is <= N/4, their sum is <= N/2.
            # This leaves large physical slack.
            pairs.append([pairable_circuits[left], pairable_circuits[right]])

            left += 1
            right -= 1

        # If one leftover pairable circuit remains, run it alone.
        if left == right:
            single_circuits_to_run_qpu.append(pairable_circuits[left])

        paired_circuits = []
        for pair in pairs:
            info = [qc[0] for qc in pair]
            meas_removed_circuits = [qc[1].remove_final_measurements(inplace=False) for qc in pair]
            total_qubits = sum([qc.num_qubits for qc in meas_removed_circuits])
            combined = QuantumCircuit(total_qubits)

            offset = 0
            for i, qc in enumerate(meas_removed_circuits):
                n = qc.num_qubits
                qargs = list(range(offset, offset + n))

                combined.compose(qc, qubits=qargs, inplace=True)

                creg = ClassicalRegister(n, f"circuit_{i}")
                combined.add_register(creg)
                combined.measure(qargs, creg)
                offset += n
            paired_circuits.append([info, combined])

        print('{} fragments are paired together into {} circuits'.format(len(pairable_circuits), len(paired_circuits)))
        print('{} fragments are run separately'.format(len(single_circuits_to_run_qpu)))
        
        # Run by chunks of max_circuits_per_job
        # 1. Single circuits
        chunk_circuits_to_run_qpu = [single_circuits_to_run_qpu[i:i + self.max_circuits_per_job] for i in range(0, len(single_circuits_to_run_qpu), self.max_circuits_per_job)]
        for chunk in chunk_circuits_to_run_qpu:
            
            # Transpile
            transpiled_qcs = [self.fragment_pm.run(k[1]) for k in chunk]

            # Run
            sampler = Sampler(mode=self.backend_fractional)
            job = sampler.run(transpiled_qcs, shots = 1)
            result = job.result()

            # Append QPU results into the appropriate noisy circuit sample
            for result_idx in range(len(transpiled_qcs)):
                bs = list(result[result_idx].data.meas.get_counts().keys())[0]
                bs = bs[::-1] # Due to changes in convention from LSB -> MSB
                p = chunk[result_idx][0]['partition']
                result_sample_idx = chunk[result_idx][0]['sample_idx']
                
                # Find corresponding noisy_circuit_sample
                for ncs_idx, ncs in enumerate(noisy_circuit_samples):
                    if ncs[0]['sample_idx'] == result_sample_idx:
                        result_ncs_idx = ncs_idx

                # Insert results into noisy_circuit_sample
                noisy_circuit_samples[result_ncs_idx][1]['|'.join([str(i) for i in list(p)])] = np.array([[int(b) for b in bs]])

        # 2. Paired circuits
        chunk_circuits_to_run_qpu = [paired_circuits[i:i + self.max_circuits_per_job] for i in range(0, len(paired_circuits), self.max_circuits_per_job)]
        for chunk in chunk_circuits_to_run_qpu:
            
            # Transpile
            transpiled_qcs = [self.fragment_pm.run(k[1]) for k in chunk]

            # Run
            sampler = Sampler(mode=self.backend_fractional)
            job = sampler.run(transpiled_qcs, shots = 1)
            result = job.result()

            # Append QPU results into the appropriate noisy circuit sample
            for result_idx in range(len(transpiled_qcs)):
                for circuit_idx in range(2):
                    bs = list(result[result_idx].data[f'circuit_{circuit_idx}'].get_counts().keys())[0]
                    bs = bs[::-1] # Due to changes in convention from LSB -> MSB
                    p = chunk[result_idx][0][circuit_idx]['partition']
                    result_sample_idx = chunk[result_idx][0][circuit_idx]['sample_idx']
                    
                    # Find corresponding noisy_circuit_sample
                    for ncs_idx, ncs in enumerate(noisy_circuit_samples):
                        if ncs[0]['sample_idx'] == result_sample_idx:
                            result_ncs_idx = ncs_idx

                    # Insert results into noisy_circuit_sample
                    noisy_circuit_samples[result_ncs_idx][1]['|'.join([str(i) for i in list(p)])] = np.array([[int(b) for b in bs]])

        # Concat
        concatenated_samples = [self._concatenate_sample(s[1], 'binary') for s in noisy_circuit_samples]

        return concatenated_samples, [bl[1] for bl in bitflip_locations_sample]