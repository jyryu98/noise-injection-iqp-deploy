import pennylane as qml
import numpy as np 
import networkx as nx
from functools import partial
from itertools import product 
import iqpopt as iqp
import jax 
import jax.numpy as jnp

    
class NoiselessNodePercolationDistributor:

    def __init__(self, G):
        """Simulate a noisy IQP circuit using smaller, less noisy quantum computers.

        Args:
            G (nx.Graph): Interaction graph of the circuit. Requires node (and edge, for mode=='iqp') attributes of name 'weight',
                which are the rotation angles of the corresponding gates.
            
        """

        self.interaction_graph = G
        self.num_qubits = len(G)
        self.q1_weights = nx.get_node_attributes(G, 'weight')
        self.q2_weights = nx.get_edge_attributes(G, 'weight')

        # Divide interaction graph into layers via edge coloring
        lineG = nx.line_graph(G)
        self.layer_map = nx.greedy_color(lineG)
        self.num_layers = max([self.layer_map[i] for i in self.layer_map.keys()])+2

        # Construct iqpopt IqpSimulator object
        self.iqpsimulator_gates = [[[i]] for i in range(self.num_qubits)]
        self.iqpsimulator_params = [-0.5*self.q1_weights[i] for i in range(self.num_qubits)]
        for edge in G.edges():
            self.iqpsimulator_gates.append([[edge[0], edge[1]]])
            self.iqpsimulator_params.append(-0.5*self.q2_weights[edge])
        self.iqpsimulator = iqp.IqpSimulator(self.num_qubits, self.iqpsimulator_gates)

        # Calculate local Pauli Z expvals via iqpopt
        n_samples = 10000
        key = jax.random.PRNGKey(42)
        self.expvals_1q, std = self.iqpsimulator.op_expval(jnp.array(self.iqpsimulator_params), jnp.eye(self.num_qubits), n_samples, key)

    def sorted_nodes(self, s):
        return sorted(list(s))

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
    
    def _sample_circuit(self, percolation_p, max_capacity, return_bitflip_locations = False):

        layerwise_percolation_p = 1-(1-percolation_p)**(1/self.num_layers)

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
        if max([len(s) for s in partition]) > max_capacity:
            for p in partition:
                noisy_circuit_sample['|'.join([str(i) for i in list(p)])] = None 
            if not return_bitflip_locations:
                return noisy_circuit_sample
            else: 
                return noisy_circuit_sample, np.zeros(self.num_qubits)

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

        # Randomize removed nodes
        for qubit_idx in removed_nodes:
            noisy_circuit_sample[str(qubit_idx)] = np.random.binomial(1, p = 0.5)

        # Simulate each cluster
        for p in partition:
            p_nodes = self.sorted_nodes(p)
            # Construct circuit for each partition larger than one qubit
            if len(p) > 1 or (len(p) == 1 and p_nodes[0] not in removed_nodes):
                dev = qml.device('default.qubit', wires = p_nodes)
                @partial(qml.set_shots, shots=1)
                @qml.qnode(device = dev)
                def partition_circuit():
                    # Single qubit gates
                    for i in p_nodes:
                        qml.Hadamard(i)
                        qml.RZ(self.q1_weights[i], i)
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
                    # Hadamard layer
                    for i in p_nodes:
                        qml.Hadamard(i)
                    return qml.sample(wires = p_nodes)
                noisy_circuit_sample['|'.join([str(i) for i in p_nodes])] = partition_circuit()
        if not return_bitflip_locations:
            return noisy_circuit_sample
        else:
            return noisy_circuit_sample, bitflip_locations

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
        
    def sample(self, num_samples, noise_p, concat_mode, max_capacity, return_fragment_sizes = False):
        """
        Sample from the noisy circuit using percolation.
        
        Args:
            num_samples (int): Number of samples
            noise_p (float): Strength of injected noise
            concat_mode (str): 'binary' or 'decimal'
            max_capacity (int): Maximum number of qubits in a cluster
            return_fragment_sizes (bool): Whether or not to return list[list[int]] of circuit fragment sizes
        Returns:
            (list[str|int]): List of samples in either bitstring or integer form
            (list[list[int]]): List of circuit fragment sizes, if return_fragment_sizes = True
        """
        noisy_circuit_samples = [self._sample_circuit(noise_p, max_capacity) for _ in range(num_samples)]
        concatenated_samples = [self._concatenate_sample(s, concat_mode) for s in noisy_circuit_samples if not any(x is None for x in list(s.values()))]
        if return_fragment_sizes:
            fragment_sizes = [[len(i.split('|')) for i in s.keys()] for s in noisy_circuit_samples]
            return concatenated_samples, fragment_sizes
        else:
            return concatenated_samples

    def sample_LocalCond_Prod_EM(self, num_samples, percolation_p, max_capacity):
        concatenated_samples = []
        noisy_circuit_bitflip_locations = []

        while len(concatenated_samples) < num_samples:
            # Sample via percolation
            ncr = self._sample_circuit(percolation_p, max_capacity, return_bitflip_locations=True)
            
            # Remove out-of-capacity shots and concatenate
            if not any(x is None for x in list(ncr[0].values())):
                noisy_circuit_bitflip_locations.append(ncr[1])
                concatenated_samples.append([int(i) for i in self._concatenate_sample(ncr[0], 'binary')])

        # Calculate local Pauli Z expvals via iqpopt
        n_samples = 10000
        key = jax.random.PRNGKey(42)
        expval, std = self.iqpsimulator.op_expval(jnp.array(self.iqpsimulator_params), jnp.eye(self.num_qubits), n_samples, key)
        expval = np.asarray(expval)
        p0s = np.clip((expval + 1.0) / 2.0, 0.0, 1.0)
        p1s = 1.0 - p0s

        prod_em_samples = []
        local_cond_em_samples = []
        for sample_idx, (cs, bl) in enumerate(zip(concatenated_samples, noisy_circuit_bitflip_locations)):
            bitflip_locations_set = set(np.where(bl == 1)[0])
            local_cond_em_sample = np.array(cs, dtype=int).copy()
            cs = np.asarray(cs, dtype=int)

            # Sample from the product distribution
            prod_distn_sample = np.random.binomial(np.ones(self.num_qubits).astype(int), p1s)

            # Replace removed node bits with the sample from the conditional circuit
            prod_em_sample = np.array(cs, dtype=int).copy()
            prod_em_sample[bl == 1] = prod_distn_sample[bl == 1]
            prod_em_samples.append(prod_em_sample)

            for node in bitflip_locations_set:
                node_neighbors_set = set([n for n in self.interaction_graph.neighbors(node)])
                valid_node_neighbors_set = node_neighbors_set - bitflip_locations_set # Remove other noisy bits

                if len(valid_node_neighbors_set) == 0:
                    # If no valid neighbors: sample from the marginal distribution
                    p1 = float(np.clip((1.0 - expval[node]) / 2.0, 0.0, 1.0))
                    correction_bit = np.random.binomial(1, p1)
                else:
                    # Sample from the local conditional distribution
                    # Note that |b><b| = (I+(-1)^b Z)/2 
                    num_valid_neighbors = len(valid_node_neighbors_set)
                    valid_neighbors_list = self.sorted_nodes(valid_node_neighbors_set)
                    local_key = jax.random.fold_in(key, sample_idx)
                    local_key = jax.random.fold_in(local_key, node)

                    denom_key = jax.random.fold_in(local_key, 0)
                    num_key = jax.random.fold_in(local_key, 1)

                    subsets = np.array(list(product([0, 1], repeat=num_valid_neighbors)), dtype=int)
                    neighbor_bits = cs[valid_neighbors_list]
                    signs = (-1) ** (subsets @ neighbor_bits)

                    operators = np.zeros((2**num_valid_neighbors, self.num_qubits), dtype=int)
                    for soi, suboperator in enumerate(product([0, 1], repeat = num_valid_neighbors)):
                        operators[soi, valid_neighbors_list] = np.array(suboperator)
                    denom_expvals, _ = self.iqpsimulator.op_expval(
                        jnp.array(self.iqpsimulator_params),
                        jnp.array(operators),
                        n_samples,
                        denom_key,
                    )
                    denom_expvals = np.asarray(denom_expvals)

                    denom = np.dot(signs, denom_expvals)

                    num_ops = operators.copy()
                    num_ops[:, node] = 1

                    num_expvals, _ = self.iqpsimulator.op_expval(
                        jnp.array(self.iqpsimulator_params),
                        jnp.array(num_ops),
                        n_samples,
                        num_key,
                    )
                    num_expvals = np.asarray(num_expvals)

                    num = np.dot(signs, num_expvals)

                    if abs(denom) < 1e-8:
                        p1 = float(np.clip((1.0 - expval[node]) / 2.0, 0.0, 1.0))
                    else:
                        z_cond = num / denom
                        p1 = float(np.clip((1.0 - z_cond) / 2.0, 0.0, 1.0))

                    correction_bit = np.random.binomial(1, p1)

                # Replace removed node bit with the local sample
                local_cond_em_sample[node] = correction_bit
            
            local_cond_em_samples.append(local_cond_em_sample)
        return local_cond_em_samples, prod_em_samples, concatenated_samples, noisy_circuit_bitflip_locations