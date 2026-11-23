import pennylane as qml
import numpy as np 
import networkx as nx
from scipy.optimize import fsolve

def solve_S_for_target_q_from_config(target_q, configuration):
    """
    Given node occupation probability and configuration, estimate the average giant cluster size.

    Args:
        target_q (float): Node occupation probability
        configuration (list): List of length (number of nodes), which indicates the degree of each node

    Returns:
        required_S (float): Estimated average giant cluster size
        u_sol (float): Estimated probability that following an edge, the connected node is in the giant cluster
    """

    avg_k = np.mean(configuration)
    min_k = np.min(configuration)
    max_k = np.max(configuration)
    ks = np.arange(min_k, max_k + 1)
    pks = np.array([np.sum(configuration == k)/len(configuration) for k in ks])
    qks = (ks * pks) / avg_k
    def G0(u): return np.sum(pks * (u**ks))
    def G1(u): return np.sum(qks * (u**(ks-1)))
    def objective(u):
        if u >= 1.0: return 1.0
        return u-(1-target_q+target_q*G1(u))
    
    u_sol = fsolve(objective, 0.5)
    required_S = target_q*(1-G0(u_sol))
    return required_S, u_sol
    
def solve_q_for_target_S_from_config(target_S, configuration):
    """
    Given the giant cluster fraction, estimate the average node occupation probability.

    Args:
        target_S (float): Giant cluster fraction
        configuration (list): List of degrees

    Returns:
        q_required (float): Required node occupation probability
        u_sol (float): Probability that following an edge, the node that is met is part of the giant cluster
    """
    avg_k = np.mean(configuration)
    min_k = np.min(configuration)
    max_k = np.max(configuration)
    ks = np.arange(min_k, max_k + 1)
    pks = np.array([np.sum(configuration == k)/len(configuration) for k in ks])
    qks = (ks * pks) / avg_k
    def G0(u): return np.sum(pks * (u**ks))
    def G1(u): return np.sum(qks * (u**(ks-1)))
    def objective(u):
        # We handle u=1 case separately as it's the trivial solution
        if u >= 1.0: return 1.0
        return u - (1 - target_S * (1 - G1(u)) / (1 - G0(u)))

    # 6. Solve for u numerically
    u_sol = fsolve(objective, 0.5)[0]
    
    # 7. Reverse engineer q: q = S / (1 - G0(u))
    q_required = target_S / (1 - G0(u_sol))
    
    return q_required, u_sol

def bitstring_samples_to_epdf(samples):
    """
    Constuct the empirical probability distribution from list of samples

    Args:
        samples (list): List of samples

    Returns:
        epdf (np.array): Empirical probability distribution
    """
    num_bits = len(samples[0])
    samples_as_int = [int(''.join([str(int(i)) for i in sample]), 2) for sample in samples]
    epdf = [np.sum(np.array(samples_as_int) == i)/len(samples) for i in range(2**num_bits)]
    return np.array(epdf)

def index_matrix_to_circuit(index_matrix, weights, bitflip_prob = 0):
    """
    Construct a Pennylane circuit given index matrix and weights

    Args:
        index_matrix (np.array): Binary matrix of size (number of gates, number of qubits), where the position of 1s indicate the support of the gate.
        weights (np.array): Vector of length (number of gates), elements are the rotation angle of the corresponding gate.
        bitflip_prob (float): Bitflip probability. Default 0
    
    Returns:
        circuit (qml.QNode)
    """

    num_nodes = index_matrix.shape[1]
    if bitflip_prob > 0:
        device = qml.device('default.mixed', num_nodes)
    else:
        device = qml.device('default.qubit', num_nodes)

    @qml.qnode(device)
    def circuit():
        for gate, weight in zip(index_matrix, weights):
            qidx = np.where(gate == 1)[0]
            qml.PauliRot(weight, len(qidx)*'X', qidx)
        if bitflip_prob > 0:
            for qidx in range(num_nodes):
                qml.PauliError("X", bitflip_prob, wires = qidx)
        return qml.probs()
    return circuit

def graph_to_circuit(graph, bitflip_prob = 0):
    num_nodes = len(graph)
    if bitflip_prob > 0:
        device = qml.device('default.mixed', num_nodes)
    else:
        device = qml.device('default.qubit', num_nodes)

    @qml.qnode(device)
    def circuit():
        # 1Q gates
        for n in graph.nodes():
            qml.RX(nx.get_node_attributes(graph, 'weight')[n], n)
        # 2Q gates
        for e in graph.edges():
            qml.IsingXX(nx.get_edge_attributes(graph, 'weight')[e], [e[0], e[1]])
        if bitflip_prob > 0:
            for qidx in range(num_nodes):
                qml.PauliError("X", bitflip_prob, wires = qidx)
        return qml.probs()
    return circuit

def graph_to_circuit_sample(graph, num_samples, bitflip_prob = 0):
    num_nodes = len(graph)
    if bitflip_prob > 0:
        device = qml.device('default.mixed', num_nodes)
    else:
        device = qml.device('default.qubit', num_nodes)
    
    @qml.set_shots(shots = num_samples)
    @qml.qnode(device)
    def circuit():
        # 1Q gates
        for n in graph.nodes():
            qml.RX(nx.get_node_attributes(graph, 'weight')[n], n)
        # 2Q gates
        for e in graph.edges():
            qml.IsingXX(nx.get_edge_attributes(graph, 'weight')[e], [e[0], e[1]])
        if bitflip_prob > 0:
            for qidx in range(num_nodes):
                qml.PauliError("X", bitflip_prob, wires = qidx)
        return qml.sample()
    return circuit

def u(S):
    Delta = ((15*S-18)/16)**2+((5*S-3)/12)**3
    return -0.5 + np.cbrt((18-15*S)/16 + np.sqrt(Delta)) + np.cbrt((18-15*S)/16 - np.sqrt(Delta))

"""
Below: Code from https://github.com/uriballo/IQP-NISQ
"""


import numpy as np
import iqpopt as iqp

def aachen_connectivity():
    """
    Creates a manually verified connectivity graph for the QPU.
    
    Returns:
        Dictionary where keys are qubit indices and values are lists of connected qubit indices.
    """
    # Initialize empty connectivity dictionary
    connectivity = {}
    
    # Initialize all qubits with empty neighbor lists
    for i in range(156):
        connectivity[i] = []
    
    # Manually add all connections
    
    # Row 0 horizontal connections (0-15)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)
    ]
    
    # Row 1 horizontal connections (20-35)
    connections.extend([
        (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
        (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35)
    ])
    
    # Row 2 horizontal connections (40-55)
    connections.extend([
        (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
        (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55)
    ])
    
    # Row 3 horizontal connections (60-75)
    connections.extend([
        (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67),
        (67, 68), (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (74, 75)
    ])
    
    # Row 4 horizontal connections (80-95)
    connections.extend([
        (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87),
        (87, 88), (88, 89), (89, 90), (90, 91), (91, 92), (92, 93), (93, 94), (94, 95)
    ])
    
    # Row 5 horizontal connections (100-115)
    connections.extend([
        (100, 101), (101, 102), (102, 103), (103, 104), (104, 105), (105, 106), (106, 107),
        (107, 108), (108, 109), (109, 110), (110, 111), (111, 112), (112, 113), (113, 114), (114, 115)
    ])
    
    # Row 6 horizontal connections (120-135)
    connections.extend([
        (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (125, 126), (126, 127),
        (127, 128), (128, 129), (129, 130), (130, 131), (131, 132), (132, 133), (133, 134), (134, 135)
    ])
    
    # Row 7 horizontal connections (140-155)
    connections.extend([
        (140, 141), (141, 142), (142, 143), (143, 144), (144, 145), (145, 146), (146, 147),
        (147, 148), (148, 149), (149, 150), (150, 151), (151, 152), (152, 153), (153, 154), (154, 155)
    ])
    
    # Vertical connections between rows
    connections.extend([
        # From row 0 to intermediate nodes
        (3, 16), (7, 17), (11, 18), (15, 19),
        # From intermediate nodes to row 1
        (16, 23), (17, 27), (18, 31), (19, 35),
        
        # From row 1 to intermediate nodes
        (21, 36), (25, 37), (29, 38), (33, 39),
        # From intermediate nodes to row 2
        (36, 41), (37, 45), (38, 49), (39, 53),
        
        # From row 2 to intermediate nodes
        (43, 56), (47, 57), (51, 58), (55, 59),
        # From intermediate nodes to row 3
        (56, 63), (57, 67), (58, 71), (59, 75),
        
        # From row 3 to intermediate nodes
        (61, 76), (65, 77), (69, 78), (73, 79),
        # From intermediate nodes to row 4
        (76, 81), (77, 85), (78, 89), (79, 93),
        
        # From row 4 to intermediate nodes
        (83, 96), (87, 97), (91, 98), (95, 99),
        # From intermediate nodes to row 5
        (96, 103), (97, 107), (98, 111), (99, 115),
        
        # From row 5 to intermediate nodes
        (101, 116), (105, 117), (109, 118), (113, 119),
        # From intermediate nodes to row 6
        (116, 121), (117, 125), (118, 129), (119, 133),
        
        # From row 6 to intermediate nodes
        (123, 136), (127, 137), (131, 138), (135, 139),
        # From intermediate nodes to row 7
        (136, 143), (137, 147), (138, 151), (139, 155)
    ])
    
    # Add all connections to the connectivity graph (bidirectional)
    for qubit1, qubit2 in connections:
        connectivity[qubit1].append(qubit2)
        connectivity[qubit2].append(qubit1)
    
    return connectivity

def efficient_connectivity_gates(
    connectivity_graph, num_positions, num_layers=1
):
    """
    Generates a gate list efficiently based on connectivity.
    Single qubit gates are generated for all positions from 0 to num_positions-1.
    Two qubit gates are generated only for connected pairs in connectivity_graph.
    Output format uses np.int64 and sorted pairs for two-qubit gates.

    :param connectivity_graph: Dictionary representing the graph connectivity.
    :param num_positions: The total number of qubit positions available.
    :param num_layers: Number of times to repeat the connectivity pattern.
    :return (list[list[list[np.int64]]]): gate list object.
    """
    final_gates_list = []

    # 1. Create single-qubit gates
    single_qubit_gates = [
        [[np.int64(i)]] for i in range(num_positions)
    ]

    # 2. Create two-qubit gates for directly connected qubits
    two_qubit_gates = []
    edges_added = set()  # To avoid adding the same edge twice (e.g., (0,1) vs (1,0))

    if connectivity_graph:
        for q1_key, neighbors in connectivity_graph.items():
            q1 = int(q1_key)
            # Ensure q1 is within the specified number of positions
            if not (0 <= q1 < num_positions):
                continue

            for q2_val in neighbors:
                q2 = int(q2_val)
                # Ensure q2 is within the specified number of positions
                if not (0 <= q2 < num_positions):
                    continue

                # Create a canonical representation of the edge (sorted tuple of ints)
                # This ensures [[0,1]] is treated the same as [[1,0]] for uniqueness
                # and that the gate itself will store the sorted pair.
                edge_tuple = tuple(sorted((q1, q2)))

                if edge_tuple not in edges_added:
                    two_qubit_gates.append(
                        [[np.int64(edge_tuple[0]), np.int64(edge_tuple[1])]]
                    )
                    edges_added.add(edge_tuple)
    
    # Sort the list of two-qubit gates for a deterministic output order.
    # This makes the output comparable to one generated by iterating combinations.
    two_qubit_gates.sort(key=lambda gate: (gate[0][0], gate[0][1]))

    # Combine single and two-qubit gates for one layer
    layer_gates = single_qubit_gates + two_qubit_gates

    # Repeat the pattern for the specified number of layers
    for _ in range(num_layers):
        final_gates_list.extend(layer_gates)

    return final_gates_list

def create_circuit(num_qubits: int, num_layers: int) -> iqp.IqpSimulator:
    """Creates an IQP circuit simulator with a fixed connectivity."""
    grid_conn = aachen_connectivity()
    gates = efficient_connectivity_gates(grid_conn, num_qubits, num_layers)
    return iqp.IqpSimulator(num_qubits, gates)

def graph_to_vec(graph: nx.Graph, nodes: int) -> np.ndarray:
    """
    Turn G (with nodes 0..nodes-1) into a flat upper-triangle bit-vector.
    
    Args:
        graph: NetworkX graph
        nodes: Number of nodes
        
    Returns:
        Flattened upper triangular adjacency matrix entries
    """
    # 1) enforce node-set
    if set(graph.nodes()) != set(range(nodes)):
        raise ValueError(
            f"Graph nodes must be exactly 0..{nodes-1}"
        )
    # 2) get adjacency (no weights), uint8
    adj = nx.to_numpy_array(
        graph,
        nodelist=range(nodes),
        dtype=np.uint8,
        weight=None
    )
    # 3) flatten strict upper-triangle
    iu = np.triu_indices(nodes, k=1)
    return adj[iu]

def vec_to_graph(vec: np.ndarray, nodes: int) -> nx.Graph:
    """
    Build an unweighted, undirected Graph with nodes 0..nodes-1
    from its strict-upper-triangle bit-vector.
    
    Args:
        vec: Flattened upper triangular vector
        nodes: Number of nodes
        
    Returns:
        NetworkX Graph
    """
    L = nodes * (nodes - 1) // 2
    if vec.size != L:
        raise ValueError(f"vec has length {vec.size}, expected {L}")
    G = nx.Graph()
    G.add_nodes_from(range(nodes))
    iu = np.triu_indices(nodes, k=1)

    edges = [
        (int(i), int(j))
        for (i, j), bit in zip(zip(iu[0], iu[1]), vec)
        if bit
    ]

    G.add_edges_from(edges)
    return G

def compute_average_density(samples):
    return np.mean([nx.density(graph) for graph in samples])

def compute_bipartite_percentage(samples):
    return np.mean([nx.is_bipartite(graph) * 100 for graph in samples])

def estrada_bipartivity(graph):
    """
    Estrada & Rodríguez-Velázquez spectral bipartivity index.
    https://doi.org/10.1103/PhysRevE.72.046105

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.
        
    Returns
    -------
    float
        Bipartivity index in [0, 1].
        1 means perfectly bipartite; smaller means less bipartite.
    """
    return nx.bipartite.spectral_bipartivity(graph)

def compute_average_bipartivity(samples):
    return np.mean([estrada_bipartivity(graph) for graph in samples]) 