from pathlib import Path
import sys

repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root))
import yaml
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd

from dist_iqp.noiseless_node import NoiselessNodePercolationDistributor
from dist_iqp.utils import solve_q_for_target_S_from_config, aachen_connectivity, efficient_connectivity_gates

best_run_details = pd.read_csv('./trained_params_iqp/sim_analysis_best_runs.csv')

for capacity in [15, 18, 21]:
    for run_idx in range(len(best_run_details)):
        model_dir = Path('./trained_params_iqp/' + '/'.join(best_run_details.iloc[run_idx]['file_path'].split('/')[2:-1]))

        # Load metadata to check the number of nodes
        config_file = model_dir / 'hyperparams.yml'

        with open(config_file, 'r') as f:
            metadata = yaml.safe_load(f)
        nodes = metadata.get('nodes')
        # Load model parameters
        params_file = model_dir / "params.npy"
        params = np.load(params_file)

        num_layers = metadata['hyperparameters']['num_layers']
        num_qubits = nodes * (nodes - 1) // 2

        grid_conn = aachen_connectivity()
        gates = efficient_connectivity_gates(grid_conn, num_qubits, num_layers)
        # Change the model into a graph
        interaction_graph = nx.Graph()
        for gate, param in zip(gates, params):
            if len(gate[0]) == 1:
                if int(gate[0][0]) in interaction_graph:
                    interaction_graph.nodes[int(gate[0][0])]['weight'] = interaction_graph.nodes[int(gate[0][0])]['weight'] -2*param # Difference in notation for angles between pennylane and iqpopt
                else:
                    interaction_graph.add_node(int(gate[0][0]), weight = -2*param) # Difference in notation for angles between pennylane and iqpopt
            elif len(gate[0]) == 2:
                if interaction_graph.has_edge(int(gate[0][0]), int(gate[0][1])):
                    interaction_graph[int(gate[0][0])][int(gate[0][1])]['weight'] = interaction_graph[int(gate[0][0])][int(gate[0][1])]['weight'] -2*param # Difference in notation for angles between pennylane and iqpopt
                else:
                    interaction_graph.add_edge(int(gate[0][0]), int(gate[0][1]), weight = -2*param) # Difference in notation for angles between pennylane and iqpopt
            else:
                raise ValueError 

        model_name = best_run_details.iloc[run_idx]['graph_type'] + '+' + best_run_details.iloc[run_idx]['density']
        degrees = [d for n, d in interaction_graph.degree()]
        q, _ = solve_q_for_target_S_from_config(capacity/28, degrees)
        sampler = NoiselessNodePercolationDistributor(interaction_graph)
        percolation_p = sampler._fine_tune_percolation_p(1-q, capacity, 2048)

        num_samples = 512

        for rep in range(50):
            p0s = (sampler.expvals_1q+1)/2
            p1s = 1-p0s
            np.save('./extensive_run_results/{}cap_{}run_{}rep_prod_base_samples'.format(capacity, run_idx, rep), np.array([np.random.binomial(np.ones(num_qubits).astype(int), p1s) for _ in range(512)]))
            local_cond_em_samples, prod_em_samples, concatenated_samples, noisy_circuit_bitflip_locations = sampler.sample_LocalCond_Prod_EM(num_samples, percolation_p, capacity)
            np.save('./extensive_run_results/{}cap_{}run_{}rep_localcond_emsamples'.format(capacity, run_idx, rep), np.array(local_cond_em_samples))
            np.save('./extensive_run_results/{}cap_{}run_{}rep_prod_emsamples'.format(capacity, run_idx, rep), np.array(prod_em_samples))
            np.save('./extensive_run_results/{}cap_{}run_{}rep_dqc_samples'.format(capacity, run_idx, rep), np.array(concatenated_samples))
            np.save('./extensive_run_results/{}cap_{}run_{}rep_bitflip_locs'.format(capacity, run_idx, rep), np.array(noisy_circuit_bitflip_locations))