from __future__ import annotations
from pathlib import Path
import sys

repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root))
import numpy as np
import networkx as nx
from itertools import combinations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
import jax
import jax.numpy as jnp
try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError(
        "This module requires JAX because IQPopt op_expval uses JAX arrays/keys."
    ) from exc

import yaml
from dist_iqp.utils import aachen_connectivity, efficient_connectivity_gates
import iqpopt as iqp

Observable = Tuple[int, ...]
ObservableBank = Dict[int, List[Observable]]

def create_circuit(num_qubits: int, num_layers: int) -> iqp.IqpSimulator:
    """Creates an IQP circuit simulator with a fixed connectivity."""
    grid_conn = aachen_connectivity()
    gates = efficient_connectivity_gates(grid_conn, num_qubits, num_layers)
    return iqp.IqpSimulator(num_qubits, gates)


def _canonical_obs(nodes):
    return tuple(sorted(int(x) for x in nodes))


def make_edge_observables(graph, max_observables=None, seed=0):
    """
    Weight-2 observables supported on interaction-graph edges.
    """
    rng = np.random.default_rng(seed)

    obs = [_canonical_obs((u, v)) for u, v in graph.edges()]
    obs = sorted(set(obs))

    if max_observables is not None and len(obs) > max_observables:
        idx = rng.choice(len(obs), size=max_observables, replace=False)
        obs = [obs[i] for i in idx]

    return obs


def make_distance2_observables(graph, max_observables=None, seed=1):
    """
    Weight-2 observables on pairs with graph distance exactly 2.

    These are not directly coupled by an RZZ gate, but they share at least
    one intermediate neighbor.
    """
    rng = np.random.default_rng(seed)
    nodes = sorted(int(x) for x in graph.nodes())

    obs = []

    for i, j in combinations(nodes, 2):
        if graph.has_edge(i, j):
            continue

        common = set(graph.neighbors(i)).intersection(graph.neighbors(j))
        if len(common) > 0:
            obs.append((i, j))

    obs = sorted(set(obs))

    if max_observables is not None and len(obs) > max_observables:
        idx = rng.choice(len(obs), size=max_observables, replace=False)
        obs = [obs[i] for i in idx]

    return obs


def make_nonlocal_pair_observables(
    graph,
    max_observables=500,
    min_distance=4,
    seed=2,
):
    """
    Weight-2 observables on graph-distant pairs.

    This is a control ensemble. Product should be competitive here.
    """
    rng = np.random.default_rng(seed)
    nodes = sorted(int(x) for x in graph.nodes())

    # Only need distances up to min_distance.
    lengths = dict(nx.all_pairs_shortest_path_length(graph, cutoff=min_distance))

    candidates = []

    for i, j in combinations(nodes, 2):
        dij = lengths.get(i, {}).get(j, np.inf)
        if dij >= min_distance:
            candidates.append((i, j))

    candidates = sorted(set(candidates))

    if len(candidates) > max_observables:
        idx = rng.choice(len(candidates), size=max_observables, replace=False)
        candidates = [candidates[i] for i in idx]

    return candidates


def make_connected_subgraph_observables(
    graph,
    weight,
    max_observables=500,
    seed=3,
    max_tries=200000,
):
    """
    Sample connected weight-k observables from the interaction graph.

    These are local graph-structured Pauli-Z supports.
    """
    if weight < 2:
        raise ValueError("Connected graph observables require weight >= 2.")

    rng = np.random.default_rng(seed)
    nodes = sorted(int(x) for x in graph.nodes())

    obs = set()
    tries = 0

    while len(obs) < max_observables and tries < max_tries:
        tries += 1

        start = int(rng.choice(nodes))
        chosen = {start}
        frontier = set(graph.neighbors(start))

        while len(chosen) < weight and len(frontier) > 0:
            new_node = int(rng.choice(sorted(frontier)))
            chosen.add(new_node)

            frontier.update(graph.neighbors(new_node))
            frontier.difference_update(chosen)

        if len(chosen) == weight:
            sub = graph.subgraph(chosen)
            if nx.is_connected(sub):
                obs.add(_canonical_obs(chosen))

    return sorted(obs)

def make_graph_informed_observable_bank(
    graph,
    weights=(2, 3, 4),
    num_observables_per_group=500,
    seed=0,
    include_edges=True,
    include_distance2=True,
    include_nonlocal=True,
    include_connected=True,
):
    """
    Build graph-informed observable groups.

    Returns
    -------
    observable_bank : dict[str, list[tuple[int]]]

    Example keys:
        edge_w2
        distance2_w2
        nonlocal_w2
        connected_w3
        connected_w4
    """
    bank = {}

    if 2 in weights:
        if include_edges:
            bank["edge_w2"] = make_edge_observables(
                graph,
                max_observables=num_observables_per_group,
                seed=seed + 10,
            )

        if include_distance2:
            bank["distance2_w2"] = make_distance2_observables(
                graph,
                max_observables=num_observables_per_group,
                seed=seed + 20,
            )

        if include_nonlocal:
            bank["nonlocal_w2"] = make_nonlocal_pair_observables(
                graph,
                max_observables=num_observables_per_group,
                min_distance=4,
                seed=seed + 30,
            )

    if include_connected:
        for w in weights:
            if w >= 3:
                bank[f"connected_w{w}"] = make_connected_subgraph_observables(
                    graph,
                    weight=w,
                    max_observables=num_observables_per_group,
                    seed=seed + 100 + w,
                )

    # Helpful sanity printout
    for name, obs in bank.items():
        print(f"{name}: {len(obs)} observables")

    return bank



def observables_to_operator_matrix(observables, num_qubits):
    """
    Convert a list of Pauli-Z supports into an iqpopt operator matrix.

    observables = [(0, 1), (3, 5, 8), ...]
    """
    ops = np.zeros((len(observables), num_qubits), dtype=int)

    for row, obs in enumerate(observables):
        for q in obs:
            ops[row, int(q)] = 1

    return ops


def compute_iqpopt_ideal_corrs_for_observable_bank(
    iqpsimulator,
    params,
    observable_bank,
    num_qubits,
    n_iqpopt_samples=10000,
    key=None,
    batch_size=None,
):
    """
    Compute ideal <Z_S> for each observable group in an arbitrary bank.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    ideal_corrs = {}
    ideal_stds = {}

    for group_idx, (group_name, observables) in enumerate(observable_bank.items()):
        if len(observables) == 0:
            ideal_corrs[group_name] = np.array([], dtype=float)
            ideal_stds[group_name] = np.array([], dtype=float)
            continue

        ops = observables_to_operator_matrix(observables, num_qubits)
        group_key = jax.random.fold_in(key, group_idx)

        if batch_size is None:
            expval, std = iqpsimulator.op_expval(
                jnp.array(params),
                jnp.array(ops),
                n_iqpopt_samples,
                group_key,
            )

            ideal_corrs[group_name] = np.asarray(expval, dtype=float)
            ideal_stds[group_name] = np.asarray(std, dtype=float)

        else:
            vals = []
            stds = []

            for start in range(0, len(ops), batch_size):
                stop = min(start + batch_size, len(ops))
                batch_key = jax.random.fold_in(group_key, start)

                expval, std = iqpsimulator.op_expval(
                    jnp.array(params),
                    jnp.array(ops[start:stop]),
                    n_iqpopt_samples,
                    batch_key,
                )

                vals.append(np.asarray(expval, dtype=float))
                stds.append(np.asarray(std, dtype=float))

            ideal_corrs[group_name] = np.concatenate(vals)
            ideal_stds[group_name] = np.concatenate(stds)

    return ideal_corrs, ideal_stds

def bit_samples_to_z_samples(samples):
    """
    bit 0 -> +1
    bit 1 -> -1
    """
    samples = np.asarray(samples, dtype=int)
    return 1 - 2 * samples


def estimate_pauli_z_correlators_from_samples(samples, observables):
    """
    Estimate <Z_S> for each observable S from bit samples.
    """
    z = bit_samples_to_z_samples(samples)
    corrs = []

    for obs in observables:
        obs = tuple(obs)

        if len(obs) == 0:
            corrs.append(1.0)
        else:
            corrs.append(np.mean(np.prod(z[:, obs], axis=1)))

    return np.asarray(corrs, dtype=float)


def ensure_run_list(samples):
    """
    Normalize samples into a list of runs.

    Accepts:
        samples.shape == (num_samples, num_qubits)
        samples.shape == (num_runs, num_samples, num_qubits)
        list of arrays
    """
    if isinstance(samples, list):
        return [np.asarray(x, dtype=int) for x in samples]

    arr = np.asarray(samples, dtype=int)

    if arr.ndim == 2:
        return [arr]

    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]

    raise ValueError("samples must be list, 2D array, or 3D array.")


def build_correlator_error_table_from_observable_bank(
    samples_by_method,
    observable_bank,
    ideal_corrs,
    metric="mae",
):
    """
    Build run-level error table for arbitrary observable groups.

    Returns dataframe with columns:
        method, run_idx, group, weight, num_observables, error
    """
    rows = []

    for method, samples in samples_by_method.items():
        run_list = ensure_run_list(samples)

        for run_idx, run_samples in enumerate(run_list):
            for group_name, observables in observable_bank.items():
                if len(observables) == 0:
                    continue

                est = estimate_pauli_z_correlators_from_samples(
                    run_samples,
                    observables,
                )

                target = np.asarray(ideal_corrs[group_name], dtype=float)
                diff = est - target

                if metric == "mae":
                    error = float(np.mean(np.abs(diff)))
                elif metric == "mse":
                    error = float(np.mean(diff**2))
                elif metric == "rmse":
                    error = float(np.sqrt(np.mean(diff**2)))
                else:
                    raise ValueError("metric must be 'mae', 'mse', or 'rmse'.")

                weights = [len(obs) for obs in observables]
                weight = int(weights[0]) if len(set(weights)) == 1 else -1

                rows.append({
                    "method": method,
                    "run_idx": run_idx,
                    "group": group_name,
                    "weight": weight,
                    "num_observables": len(observables),
                    "error": error,
                })

    return pd.DataFrame(rows)

def bootstrap_vs_product_by_group(
    df,
    product_method="Product",
    num_bootstrap=5000,
    seed=0,
):
    """
    Paired bootstrap:

        delta = Err(method) - Err(Product)

    Negative delta means method beats Product.
    """
    rng = np.random.default_rng(seed)
    rows = []

    methods = [m for m in sorted(df["method"].unique()) if m != product_method]

    for group_name in sorted(df["group"].unique()):
        for method in methods:
            a = df[(df["method"] == method) & (df["group"] == group_name)]
            p = df[(df["method"] == product_method) & (df["group"] == group_name)]

            merged = a.merge(
                p,
                on=["run_idx", "group"],
                suffixes=("_method", "_product"),
            )

            if len(merged) == 0:
                continue

            delta = (
                merged["error_method"].to_numpy(dtype=float)
                - merged["error_product"].to_numpy(dtype=float)
            )

            n = len(delta)
            boot = np.empty(num_bootstrap, dtype=float)

            for b in range(num_bootstrap):
                idx = rng.integers(0, n, size=n)
                boot[b] = np.mean(delta[idx])

            rows.append({
                "group": group_name,
                "weight": int(merged["weight_method"].iloc[0]),
                "method": method,
                "num_pairs": n,
                "delta_mean": float(np.mean(delta)),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "p_delta_less_than_0": float(np.mean(boot < 0)),
                "p_delta_greater_than_0": float(np.mean(boot > 0)),
            })

    return pd.DataFrame(rows)


def bootstrap_vs_product_by_group_across_models(
    df,
    product_method="Product",
    num_bootstrap=5000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    rows = []

    methods = [m for m in sorted(df["method"].unique()) if m != product_method]

    for group_name in sorted(df["group"].unique()):
        for method in methods:
            a = df[(df["method"] == method) & (df["group"] == group_name)]
            p = df[(df["method"] == product_method) & (df["group"] == group_name)]

            merged = a.merge(
                p,
                on=["model", "run_idx", "group"],
                suffixes=("_method", "_product"),
            )

            if len(merged) == 0:
                continue

            delta = (
                merged["error_method"].to_numpy(dtype=float)
                - merged["error_product"].to_numpy(dtype=float)
            )

            n = len(delta)
            boot = np.empty(num_bootstrap, dtype=float)

            for b in range(num_bootstrap):
                idx = rng.integers(0, n, size=n)
                boot[b] = np.mean(delta[idx])

            rows.append({
                "group": group_name,
                "weight": int(merged["weight_method"].iloc[0]),
                "method": method,
                "num_pairs": n,
                "num_models": int(merged["model"].nunique()),
                "delta_mean": float(np.mean(delta)),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "p_delta_less_than_0": float(np.mean(boot < 0)),
                "p_delta_greater_than_0": float(np.mean(boot > 0)),
            })

    return pd.DataFrame(rows)

def run_graph_informed_correlator_comparison_with_iqpopt(
    samples_by_method,
    interaction_graph,
    iqpsimulator,
    params,
    num_qubits,
    weights=(2, 3, 4),
    num_observables_per_group=500,
    observable_seed=0,
    n_iqpopt_samples=10000,
    iqpopt_seed=0,
    batch_size=None,
    metric="mae",
):
    """
    Graph-informed correlator analysis for one circuit/model.

    Returns
    -------
    df : run-level error dataframe
    bootstrap_df : paired bootstrap vs Product
    observable_bank : graph-informed observable bank
    ideal_corrs : ideal iqpopt correlators
    ideal_stds : iqpopt standard deviations
    """
    observable_bank = make_graph_informed_observable_bank(
        graph=interaction_graph,
        weights=weights,
        num_observables_per_group=num_observables_per_group,
        seed=observable_seed,
    )

    ideal_corrs, ideal_stds = compute_iqpopt_ideal_corrs_for_observable_bank(
        iqpsimulator=iqpsimulator,
        params=params,
        observable_bank=observable_bank,
        num_qubits=num_qubits,
        n_iqpopt_samples=n_iqpopt_samples,
        key=jax.random.PRNGKey(iqpopt_seed),
        batch_size=batch_size,
    )

    df = build_correlator_error_table_from_observable_bank(
        samples_by_method=samples_by_method,
        observable_bank=observable_bank,
        ideal_corrs=ideal_corrs,
        metric=metric,
    )

    bootstrap_df = bootstrap_vs_product_by_group(
        df,
        product_method="Product",
        num_bootstrap=5000,
        seed=0,
    )

    return df, bootstrap_df, observable_bank, ideal_corrs, ideal_stds

def run_graph_informed_correlator_comparison_for_models(
    model_data,
    weights=(2, 3, 4),
    num_observables_per_group=500,
    observable_seed=0,
    n_iqpopt_samples=10000,
    iqpopt_seed=0,
    batch_size=None,
    metric="mae",
):
    """
    model_data[model_name] should contain:
        graph
        iqpsimulator
        params
        num_qubits
        samples_by_method
    """
    all_dfs = []
    observable_banks = {}
    ideal_corrs_by_model = {}
    ideal_stds_by_model = {}

    for model_idx, (model_name, item) in enumerate(model_data.items()):
        print(f"Running graph-informed correlator analysis for {model_name}")

        df, _, observable_bank, ideal_corrs, ideal_stds = (
            run_graph_informed_correlator_comparison_with_iqpopt(
                samples_by_method=item["samples_by_method"],
                interaction_graph=item["graph"],
                iqpsimulator=item["iqpsimulator"],
                params=item["params"],
                num_qubits=item["num_qubits"],
                weights=weights,
                num_observables_per_group=num_observables_per_group,
                observable_seed=observable_seed + 1000 * model_idx,
                n_iqpopt_samples=n_iqpopt_samples,
                iqpopt_seed=iqpopt_seed + 1000 * model_idx,
                batch_size=batch_size,
                metric=metric,
            )
        )

        df = df.copy()
        df["model"] = model_name
        all_dfs.append(df)

        observable_banks[model_name] = observable_bank
        ideal_corrs_by_model[model_name] = ideal_corrs
        ideal_stds_by_model[model_name] = ideal_stds

    all_df = pd.concat(all_dfs, ignore_index=True)

    bootstrap_df = bootstrap_vs_product_by_group_across_models(
        all_df,
        product_method="Product",
        num_bootstrap=5000,
        seed=0,
    )

    return all_df, bootstrap_df, observable_banks, ideal_corrs_by_model, ideal_stds_by_model

if __name__ == "__main__":
    # Example skeleton:
    #
    best_run_details = pd.read_csv('./trained_params_iqp/sim_analysis_best_runs.csv')
    methods = [
        "Product",
        "DQC+ProdEM_m15",
        "DQC+ProdEM_m18",
        "DQC+ProdEM_m21",
        "DQC+LocalCondEM_m15",
        "DQC+LocalCondEM_m18",
        "DQC+LocalCondEM_m21",
        "DQC_m15",
        "DQC_m18",
        "DQC_m21"
    ]
    samples = {}
    iqpopt_objects = {}
    for run_idx in range(len(best_run_details)):
        run_name = best_run_details.iloc[run_idx]['graph_type'].lower() + '_' + best_run_details.iloc[run_idx]['density'].lower()
        samples[run_name] = {}
        samples[run_name]['samples_by_method'] = {sm: [] for sm in methods}

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
        circuit = create_circuit(num_qubits, num_layers)
    
        n_samples = 10000
        key = jax.random.PRNGKey(42)
        expval, std = circuit.op_expval(params, jnp.eye(num_qubits), n_samples, key)
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
    
        p0s = (expval+1)/2
        p1s = 1-p0s

        for rep in range(50):
            for capacity in [15, 18, 21]:
                samples[run_name]['samples_by_method']["Product"].append(np.load('./extensive_run_results/{}cap_{}run_{}rep_prod_base_samples.npy'.format(capacity, run_idx, rep)))
                samples[run_name]['samples_by_method']["DQC+ProdEM_m{}".format(capacity)].append(np.load('./extensive_run_results/{}cap_{}run_{}rep_prod_emsamples.npy'.format(capacity, run_idx, rep)))
                samples[run_name]['samples_by_method']["DQC+LocalCondEM_m{}".format(capacity)].append(np.load('./extensive_run_results/{}cap_{}run_{}rep_localcond_emsamples.npy'.format(capacity, run_idx, rep)))
                samples[run_name]['samples_by_method']["DQC_m{}".format(capacity)].append(np.load('./extensive_run_results/{}cap_{}run_{}rep_dqc_samples.npy'.format(capacity, run_idx, rep)))
        samples[run_name]['iqpsimulator'] = circuit
        samples[run_name]['graph'] = interaction_graph
        samples[run_name]['params'] = params 
        samples[run_name]['num_qubits'] = num_qubits

        iqpopt_objects[run_name] = {"simulator": circuit, "params": params}

    
    all_df, bootstrap_df, observable_banks, ideal_corrs_by_model, ideal_stds_by_model = run_graph_informed_correlator_comparison_for_models(
        samples,
        weights=(2, 3, 4),
        num_observables_per_group=500,
        observable_seed=0,
        n_iqpopt_samples=10000,
        iqpopt_seed=0,
        batch_size=None,
        metric="mae",
    )

    all_df.to_csv('graph_correlator_comparison.csv')
