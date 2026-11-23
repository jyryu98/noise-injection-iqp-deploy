import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

import jax
import jax.numpy as jnp


# ============================================================
# Observable construction
# ============================================================

def canonical_obs(nodes):
    return tuple(sorted(int(x) for x in nodes))


def make_one_body_observables(num_qubits):
    return [(i,) for i in range(num_qubits)]


def make_random_weight_observables(
    num_qubits,
    weight,
    num_observables=1000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    obs = set()

    # If asking for all possible observables for small cases.
    max_possible = None
    if weight == 1:
        max_possible = num_qubits
    elif weight == 2:
        max_possible = num_qubits * (num_qubits - 1) // 2

    if max_possible is not None and num_observables >= max_possible:
        return [canonical_obs(x) for x in combinations(range(num_qubits), weight)]

    while len(obs) < num_observables:
        support = rng.choice(num_qubits, size=weight, replace=False)
        obs.add(canonical_obs(support))

    return sorted(obs)


def make_edge_observables(graph):
    return sorted({canonical_obs((u, v)) for u, v in graph.edges()})


def make_distance2_observables(
    graph,
    max_observables=1000,
    seed=1,
):
    rng = np.random.default_rng(seed)
    nodes = sorted(int(x) for x in graph.nodes())

    obs = []
    for i, j in combinations(nodes, 2):
        if graph.has_edge(i, j):
            continue

        common_neighbors = set(graph.neighbors(i)).intersection(graph.neighbors(j))
        if len(common_neighbors) > 0:
            obs.append((i, j))

    obs = sorted(set(obs))

    if max_observables is not None and len(obs) > max_observables:
        idx = rng.choice(len(obs), size=max_observables, replace=False)
        obs = [obs[i] for i in idx]

    return obs


def make_nonlocal_pair_observables(
    graph,
    min_distance=4,
    max_observables=1000,
    seed=2,
):
    rng = np.random.default_rng(seed)
    nodes = sorted(int(x) for x in graph.nodes())

    lengths = dict(nx.all_pairs_shortest_path_length(graph, cutoff=min_distance))

    obs = []
    for i, j in combinations(nodes, 2):
        dij = lengths.get(i, {}).get(j, np.inf)
        if dij >= min_distance:
            obs.append((i, j))

    obs = sorted(set(obs))

    if max_observables is not None and len(obs) > max_observables:
        idx = rng.choice(len(obs), size=max_observables, replace=False)
        obs = [obs[i] for i in idx]

    return obs


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

    if max_observables < 0:
        raise ValueError("max_observables must be nonnegative.")
    if max_tries < 0:
        raise ValueError("max_tries must be nonnegative.")
    if graph.number_of_nodes() < weight or max_observables == 0:
        return []
    if not any(len(c) >= weight for c in nx.connected_components(graph)):
        return []

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
                obs.add(canonical_obs(chosen))

    return sorted(obs)



def make_ibm_correlator_bank(
    num_qubits,
    graph=None,
    num_random_pairs=1000,
    num_distance2_pairs=1000,
    num_nonlocal_pairs=1000,
    include_random_w3=True,
    num_random_w3=1000,
    seed=0,
    *,
    include_connected_w3=True,
    include_connected_w4=True,
    num_connected_w3=500,
    num_connected_w4=500,
    connected_max_tries=200000,
):
    """
    Build observable groups for IBM hardware correlator diagnostics.

    With graph supplied, connected_w3 and connected_w4 are enabled by default.
    Each support induces a connected subgraph, using the same frontier-growth
    sampling as graph_correlator_comparison_iqpopt.py. Counts are upper limits:
    fewer supports may be returned if the graph or attempt limit restricts them.
    Without graph, only the original non-graph families are constructed.
    """
    bank = {}

    bank["one_body"] = make_one_body_observables(num_qubits)

    bank["random_w2"] = make_random_weight_observables(
        num_qubits=num_qubits,
        weight=2,
        num_observables=num_random_pairs,
        seed=seed + 10,
    )

    if include_random_w3:
        bank["random_w3"] = make_random_weight_observables(
            num_qubits=num_qubits,
            weight=3,
            num_observables=num_random_w3,
            seed=seed + 20,
        )

    if graph is not None:
        bank["edge_w2"] = make_edge_observables(graph)

        bank["distance2_w2"] = make_distance2_observables(
            graph,
            max_observables=num_distance2_pairs,
            seed=seed + 30,
        )

        bank["nonlocal_w2"] = make_nonlocal_pair_observables(
            graph,
            min_distance=4,
            max_observables=num_nonlocal_pairs,
            seed=seed + 40,
        )

        for weight, include, count in (
            (3, include_connected_w3, num_connected_w3),
            (4, include_connected_w4, num_connected_w4),
        ):
            if include:
                bank[f"connected_w{weight}"] = make_connected_subgraph_observables(
                    graph,
                    weight=weight,
                    max_observables=count,
                    seed=seed + 100 + weight,
                    max_tries=connected_max_tries,
                )

    for name, obs in bank.items():
        print(f"{name}: {len(obs)} observables")

    return bank


# ============================================================
# Correlator estimation
# ============================================================

def bit_samples_to_z_samples(samples):
    """
    bit 0 -> +1
    bit 1 -> -1
    """
    samples = np.asarray(samples, dtype=np.int8)
    return 1 - 2 * samples


def estimate_pauli_z_correlators_from_samples(samples, observables):
    """
    Estimate <Z_S> for each support S.

    samples: shape (num_samples, num_qubits)
    observables: list of tuples, e.g. [(0,), (0, 3), (1, 4, 8)]
    """
    samples = np.asarray(samples, dtype=np.int8)
    z = bit_samples_to_z_samples(samples)

    corrs = np.empty(len(observables), dtype=float)

    for k, obs in enumerate(observables):
        obs = tuple(obs)

        if len(obs) == 0:
            corrs[k] = 1.0
        elif len(obs) == 1:
            corrs[k] = np.mean(z[:, obs[0]])
        else:
            corrs[k] = np.mean(np.prod(z[:, obs], axis=1))

    return corrs


def observables_to_operator_matrix(observables, num_qubits):
    """
    Convert supports into IQPopt operator matrix.
    """
    ops = np.zeros((len(observables), num_qubits), dtype=np.int8)

    for row, obs in enumerate(observables):
        for q in obs:
            ops[row, int(q)] = 1

    return ops


def compute_iqpopt_ideal_corrs_for_bank(
    iqpsimulator,
    params,
    observable_bank,
    num_qubits,
    n_iqpopt_samples=10000,
    key=None,
    batch_size=500,
):
    """
    Compute ideal <Z_S> values for each observable group using IQPopt.

    Assumes:
        iqpsimulator.op_expval(params, ops, n_samples, key)
    returns:
        expval, std
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    ideal_corrs = {}
    ideal_stds = {}

    params_jax = jnp.array(params)

    for group_idx, (group_name, observables) in enumerate(observable_bank.items()):
        if len(observables) == 0:
            ideal_corrs[group_name] = np.array([], dtype=float)
            ideal_stds[group_name] = np.array([], dtype=float)
            continue

        ops = observables_to_operator_matrix(observables, num_qubits)
        group_key = jax.random.fold_in(key, group_idx)

        vals = []
        stds = []

        for start in range(0, len(ops), batch_size):
            stop = min(start + batch_size, len(ops))
            batch_key = jax.random.fold_in(group_key, start)

            expval, std = iqpsimulator.op_expval(
                params_jax,
                jnp.array(ops[start:stop]),
                n_iqpopt_samples,
                batch_key,
            )

            vals.append(np.asarray(expval, dtype=float))
            stds.append(np.asarray(std, dtype=float))

        ideal_corrs[group_name] = np.concatenate(vals)
        ideal_stds[group_name] = np.concatenate(stds)

    return ideal_corrs, ideal_stds


# ============================================================
# Error computation
# ============================================================

def correlator_group_error(samples, observables, ideal_corrs, metric="mae"):
    """
    Compute observable-group error for one sample set.
    """
    est = estimate_pauli_z_correlators_from_samples(samples, observables)
    ideal_corrs = np.asarray(ideal_corrs, dtype=float)

    diff = est - ideal_corrs

    if metric == "mae":
        return float(np.mean(np.abs(diff)))
    elif metric == "mse":
        return float(np.mean(diff**2))
    elif metric == "rmse":
        return float(np.sqrt(np.mean(diff**2)))
    else:
        raise ValueError("metric must be 'mae', 'mse', or 'rmse'.")


def compute_point_error_table_single_set(
    samples_by_method,
    observable_bank,
    ideal_corrs,
    metric="mae",
):
    """
    Point estimates only. One row per method and observable group.
    """
    rows = []

    for method, samples in samples_by_method.items():
        samples = np.asarray(samples, dtype=np.int8)

        for group_name, observables in observable_bank.items():
            err = correlator_group_error(
                samples=samples,
                observables=observables,
                ideal_corrs=ideal_corrs[group_name],
                metric=metric,
            )

            weights = [len(obs) for obs in observables]
            weight = int(weights[0]) if len(set(weights)) == 1 else -1

            rows.append({
                "method": method,
                "group": group_name,
                "weight": weight,
                "num_samples": int(samples.shape[0]),
                "num_observables": int(len(observables)),
                "error": err,
            })

    return pd.DataFrame(rows)


# ============================================================
# Measurement-sample bootstrap
# ============================================================

def bootstrap_error_single_method(
    samples,
    observables,
    ideal_corrs,
    metric="mae",
    num_bootstrap=5000,
    seed=0,
):
    """
    Bootstrap finite-measurement-sample uncertainty for one method.

    Each bootstrap replicate resamples bitstrings with replacement.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.int8)
    n = samples.shape[0]

    point_error = correlator_group_error(
        samples=samples,
        observables=observables,
        ideal_corrs=ideal_corrs,
        metric=metric,
    )

    boot_errors = np.empty(num_bootstrap, dtype=float)

    for b in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_samples = samples[idx]

        boot_errors[b] = correlator_group_error(
            samples=boot_samples,
            observables=observables,
            ideal_corrs=ideal_corrs,
            metric=metric,
        )

    return {
        "error_mean": point_error,
        "ci95_low": float(np.quantile(boot_errors, 0.025)),
        "ci95_high": float(np.quantile(boot_errors, 0.975)),
    }


def bootstrap_error_difference_two_methods(
    samples_a,
    samples_b,
    observables,
    ideal_corrs,
    metric="mae",
    num_bootstrap=5000,
    seed=0,
    paired=False,
):
    """
    Bootstrap difference:

        delta = error(method_a) - error(method_b)

    Negative delta means method_a has lower error.

    paired=True only makes sense if row i of samples_a and row i of samples_b
    correspond to the same deployment instance / percolation mask.
    """
    rng = np.random.default_rng(seed)

    samples_a = np.asarray(samples_a, dtype=np.int8)
    samples_b = np.asarray(samples_b, dtype=np.int8)

    n_a = samples_a.shape[0]
    n_b = samples_b.shape[0]

    point_a = correlator_group_error(
        samples=samples_a,
        observables=observables,
        ideal_corrs=ideal_corrs,
        metric=metric,
    )

    point_b = correlator_group_error(
        samples=samples_b,
        observables=observables,
        ideal_corrs=ideal_corrs,
        metric=metric,
    )

    point_delta = point_a - point_b

    boot_delta = np.empty(num_bootstrap, dtype=float)

    for k in range(num_bootstrap):
        if paired:
            if n_a != n_b:
                raise ValueError("paired=True requires equal sample sizes.")

            idx = rng.integers(0, n_a, size=n_a)
            boot_a = samples_a[idx]
            boot_b = samples_b[idx]

        else:
            idx_a = rng.integers(0, n_a, size=n_a)
            idx_b = rng.integers(0, n_b, size=n_b)
            boot_a = samples_a[idx_a]
            boot_b = samples_b[idx_b]

        err_a = correlator_group_error(
            samples=boot_a,
            observables=observables,
            ideal_corrs=ideal_corrs,
            metric=metric,
        )

        err_b = correlator_group_error(
            samples=boot_b,
            observables=observables,
            ideal_corrs=ideal_corrs,
            metric=metric,
        )

        boot_delta[k] = err_a - err_b

    boot_delta_mean = float(np.mean(boot_delta))
    centered = boot_delta - boot_delta_mean

    return {
        "error_a": point_a,
        "error_b": point_b,

        # Observed point estimate
        "delta_point": point_delta,

        # Bootstrap distribution summary
        "delta_boot_mean": boot_delta_mean,
        "ci95_low_raw": float(np.quantile(boot_delta, 0.025)),
        "ci95_high_raw": float(np.quantile(boot_delta, 0.975)),

        # Recommended CI for reporting with delta_point
        "ci95_low_centered": float(point_delta + np.quantile(centered, 0.025)),
        "ci95_high_centered": float(point_delta + np.quantile(centered, 0.975)),

        "p_delta_less_than_0_raw": float(np.mean(boot_delta < 0)),
        "p_delta_greater_than_0_raw": float(np.mean(boot_delta > 0)),
    }


def summarize_errors_with_sample_bootstrap(
    samples_by_method,
    observable_bank,
    ideal_corrs,
    metric="mae",
    num_bootstrap=5000,
    seed=0,
):
    """
    Bootstrap CI for each method's error.
    """
    rows = []

    for method_idx, (method, samples) in enumerate(samples_by_method.items()):
        samples = np.asarray(samples, dtype=np.int8)

        for group_idx, (group_name, observables) in enumerate(observable_bank.items()):
            stats = bootstrap_error_single_method(
                samples=samples,
                observables=observables,
                ideal_corrs=ideal_corrs[group_name],
                metric=metric,
                num_bootstrap=num_bootstrap,
                seed=seed + 1000 * method_idx + group_idx,
            )

            weights = [len(obs) for obs in observables]
            weight = int(weights[0]) if len(set(weights)) == 1 else -1

            rows.append({
                "method": method,
                "group": group_name,
                "weight": weight,
                "num_samples": int(samples.shape[0]),
                "num_observables": int(len(observables)),
                **stats,
            })

    return pd.DataFrame(rows)


def compare_methods_with_sample_bootstrap(
    samples_by_method,
    observable_bank,
    ideal_corrs,
    baseline_method,
    metric="mae",
    num_bootstrap=5000,
    seed=0,
    paired=False,
):
    """
    Compare every non-baseline method against baseline_method.

    delta = error(method) - error(baseline)

    Negative delta means method is better than baseline.
    """
    if baseline_method not in samples_by_method:
        raise ValueError(f"{baseline_method=} not found in samples_by_method.")

    rows = []
    baseline_samples = samples_by_method[baseline_method]

    methods = [m for m in samples_by_method if m != baseline_method]

    for method_idx, method in enumerate(methods):
        method_samples = samples_by_method[method]

        for group_idx, (group_name, observables) in enumerate(observable_bank.items()):
            stats = bootstrap_error_difference_two_methods(
                samples_a=method_samples,
                samples_b=baseline_samples,
                observables=observables,
                ideal_corrs=ideal_corrs[group_name],
                metric=metric,
                num_bootstrap=num_bootstrap,
                seed=seed + 1000 * method_idx + group_idx,
                paired=paired,
            )

            weights = [len(obs) for obs in observables]
            weight = int(weights[0]) if len(set(weights)) == 1 else -1

            rows.append({
                "method": method,
                "baseline": baseline_method,
                "group": group_name,
                "weight": weight,
                "num_samples_method": int(np.asarray(method_samples).shape[0]),
                "num_samples_baseline": int(np.asarray(baseline_samples).shape[0]),
                "num_observables": int(len(observables)),
                **stats,
            })

    return pd.DataFrame(rows)


# ============================================================
# Full IBM diagnostic wrapper
# ============================================================

def run_ibm_correlator_diagnostics_single_sample_set(
    samples_by_method,
    num_qubits,
    iqpsimulator,
    params,
    graph=None,
    num_random_pairs=1000,
    num_distance2_pairs=1000,
    num_nonlocal_pairs=1000,
    include_random_w3=True,
    num_random_w3=1000,
    observable_seed=0,
    n_iqpopt_samples=10000,
    iqpopt_seed=0,
    batch_size=500,
    metric="mae",
    num_bootstrap=5000,
    paired_vs_product=False,
    paired_vs_protocol=True,
    *,
    include_connected_w3=True,
    include_connected_w4=True,
    num_connected_w3=500,
    num_connected_w4=500,
    connected_max_tries=200000,
):
    """
    Full IBM correlator diagnostic pipeline for one sample set per method.

    Connected weight-3/4 groups are enabled when graph is supplied. Control
    them with include_connected_w3/w4 and num_connected_w3/w4; the counts
    are upper limits. All returned error tables include the enabled groups.

    Returns
    -------
    point_df:
        Point estimates without bootstrap.
    summary_df:
        Measurement-sample bootstrap CI for each method's error.
    vs_product_df:
        Bootstrap difference against Product.
    vs_protocol_df:
        Bootstrap difference against Protocol.
    observable_bank:
        Dict of observable groups.
    ideal_corrs:
        IQPopt ideal correlators.
    ideal_stds:
        IQPopt stds.
    """
    observable_bank = make_ibm_correlator_bank(
        num_qubits=num_qubits,
        graph=graph,
        num_random_pairs=num_random_pairs,
        num_distance2_pairs=num_distance2_pairs,
        num_nonlocal_pairs=num_nonlocal_pairs,
        include_random_w3=include_random_w3,
        num_random_w3=num_random_w3,
        seed=observable_seed,
        include_connected_w3=include_connected_w3,
        include_connected_w4=include_connected_w4,
        num_connected_w3=num_connected_w3,
        num_connected_w4=num_connected_w4,
        connected_max_tries=connected_max_tries,
    )

    ideal_corrs, ideal_stds = compute_iqpopt_ideal_corrs_for_bank(
        iqpsimulator=iqpsimulator,
        params=params,
        observable_bank=observable_bank,
        num_qubits=num_qubits,
        n_iqpopt_samples=n_iqpopt_samples,
        key=jax.random.PRNGKey(iqpopt_seed),
        batch_size=batch_size,
    )

    point_df = compute_point_error_table_single_set(
        samples_by_method=samples_by_method,
        observable_bank=observable_bank,
        ideal_corrs=ideal_corrs,
        metric=metric,
    )

    summary_df = summarize_errors_with_sample_bootstrap(
        samples_by_method=samples_by_method,
        observable_bank=observable_bank,
        ideal_corrs=ideal_corrs,
        metric=metric,
        num_bootstrap=num_bootstrap,
        seed=observable_seed + 10_000,
    )

    if "Product" in samples_by_method:
        vs_product_df = compare_methods_with_sample_bootstrap(
            samples_by_method=samples_by_method,
            observable_bank=observable_bank,
            ideal_corrs=ideal_corrs,
            baseline_method="Product",
            metric=metric,
            num_bootstrap=num_bootstrap,
            seed=observable_seed + 20_000,
            paired=paired_vs_product,
        )
    else:
        vs_product_df = pd.DataFrame()

    if "Protocol" in samples_by_method:
        vs_protocol_df = compare_methods_with_sample_bootstrap(
            samples_by_method=samples_by_method,
            observable_bank=observable_bank,
            ideal_corrs=ideal_corrs,
            baseline_method="Protocol",
            metric=metric,
            num_bootstrap=num_bootstrap,
            seed=observable_seed + 30_000,
            paired=paired_vs_protocol,
        )
    else:
        vs_protocol_df = pd.DataFrame()

    return (
        point_df,
        summary_df,
        vs_product_df,
        vs_protocol_df,
        observable_bank,
        ideal_corrs,
        ideal_stds,
    )

import json
from ast import literal_eval
import iqpopt as iqp

if __name__ == '__main__':
    samples_by_method = {
        "Product": np.load('./ibm_results/prod_baseline_samples.npy'),
        "Protocol": np.load('./ibm_results/percolation_samples.npy'),
        "Protocol+CondIQPEM": np.load('./ibm_results/localcondem_samples.npy'),
        "Protocol+ProdEM": np.load('./ibm_results/prodem_samples.npy'),
    }
    
    with open('./ibm_results/large_circuit_rot_angles.json', 'r') as f:
        rot_angles = json.load(f)

    # Build IQPopt simulator
    gates = []
    params = []
    num_qubits = 0

    for k, angle_raw in rot_angles.items():
        angle = angle_raw / 2
        params.append(angle)

        gate = literal_eval(k)

        if isinstance(gate, int):
            qubits = [gate]
        else:
            qubits = list(gate)

        gates.append([qubits])
        num_qubits = max(num_qubits, max(qubits) + 1)

    iqpsimulator = iqp.IqpSimulator(num_qubits, gates)
    iqpsimulator_params = np.array(params)

    with open('./noise_fitting_results/layer_map.json', 'r') as f:
        layer_map = json.load(f)

    interaction_graph = nx.Graph()
    interaction_graph.add_nodes_from(range(316))
    edges = [literal_eval(k) for k in layer_map.keys()] + [(141, 156), (145, 157), (149, 158), (153, 159), (156, 161), (157, 165), (158, 169), (159, 173)] + [(literal_eval(k)[0]+160, literal_eval(k)[1]+160) for k in layer_map.keys()]
    interaction_graph.add_edges_from(edges)

    (
        point_df,
        summary_df,
        vs_product_df,
        vs_protocol_df,
        observable_bank,
        ideal_corrs,
        ideal_stds,
    ) = run_ibm_correlator_diagnostics_single_sample_set(
        samples_by_method=samples_by_method,
        num_qubits=316,
        iqpsimulator=iqpsimulator,
        params=iqpsimulator_params,
        graph=interaction_graph,
        num_random_pairs=1000,
        num_distance2_pairs=1000,
        num_nonlocal_pairs=1000,
        include_random_w3=False, ##################################
        num_random_w3=1000,
        include_connected_w3=True,
        include_connected_w4=True,
        num_connected_w3=1000,
        num_connected_w4=1000,
        observable_seed=123,
        n_iqpopt_samples=10000,
        iqpopt_seed=0,
        batch_size=500,
        metric="mae",
        num_bootstrap=5000, ##################################

        # Product samples are usually not paired with hardware samples.
        paired_vs_product=False,

        # Set True only if Protocol and Protocol+EM sample i correspond
        # to the same percolation/deployment instance.
        paired_vs_protocol=True,
    )

    print("Point estimates")
    print(point_df)

    print("Bootstrap summary")
    print(summary_df)

    print("Compared against Product")
    print(vs_product_df)

    print("Compared against Protocol")
    print(vs_protocol_df)

    point_df.to_csv("./ibm_results/ibm_correlator_point_estimates.csv", index=False)
    summary_df.to_csv("./ibm_results/ibm_correlator_summary_sample_bootstrap.csv", index=False)
    vs_product_df.to_csv("./ibm_results/ibm_correlator_vs_product_sample_bootstrap.csv", index=False)
    vs_protocol_df.to_csv("./ibm_results/ibm_correlator_vs_protocol_sample_bootstrap.csv", index=False)
