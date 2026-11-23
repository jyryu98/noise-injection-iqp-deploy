from __future__ import annotations
from pathlib import Path
import sys

repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root))
import numpy as np
import networkx as nx

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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

# ---------------------------------------------------------------------------
# Basic observable/correlator utilities
# ---------------------------------------------------------------------------

def bit_samples_to_z_samples(samples: np.ndarray) -> np.ndarray:
    """Convert bit samples x in {0, 1} to Pauli-Z samples z in {+1, -1}."""
    samples = np.asarray(samples, dtype=np.int8)
    if samples.ndim != 2:
        raise ValueError(f"samples must have shape (num_samples, num_qubits), got {samples.shape}")
    return 1 - 2 * samples


def random_observables(
    num_qubits: int,
    weight: int,
    num_observables: int,
    rng: np.random.Generator,
) -> List[Observable]:
    """Generate random distinct Pauli-Z observable supports of fixed weight."""
    if weight < 1:
        raise ValueError("weight must be at least 1.")
    if weight > num_qubits:
        raise ValueError("weight cannot exceed num_qubits.")

    max_possible = math.comb(num_qubits, weight)
    num_observables = min(int(num_observables), max_possible)

    observables = set()
    while len(observables) < num_observables:
        obs = tuple(sorted(rng.choice(num_qubits, size=weight, replace=False)))
        observables.add(obs)

    return sorted(observables)


def make_observable_bank(
    num_qubits: int,
    weights: Sequence[int] = (1, 2, 3, 4),
    num_observables_per_weight: int = 500,
    seed: int = 1234,
) -> ObservableBank:
    """
    Make one shared random observable bank.

    Use the same bank for all methods/runs of the same model.
    """
    rng = np.random.default_rng(seed)
    bank: ObservableBank = {}

    for w in weights:
        bank[int(w)] = random_observables(
            num_qubits=num_qubits,
            weight=int(w),
            num_observables=num_observables_per_weight,
            rng=rng,
        )

    return bank


def observables_to_operator_matrix(
    observables: Sequence[Observable],
    num_qubits: int,
    dtype=np.int8,
) -> np.ndarray:
    """
    Convert observable supports into a binary operator matrix.

    operators[k, i] = 1 if observable k includes Z_i.
    """
    operators = np.zeros((len(observables), num_qubits), dtype=dtype)

    for row, obs in enumerate(observables):
        obs = tuple(obs)

        if len(obs) == 0:
            continue

        if min(obs) < 0 or max(obs) >= num_qubits:
            raise ValueError(f"Observable {obs} outside num_qubits={num_qubits}.")

        operators[row, list(obs)] = 1

    return operators


def estimate_correlators_from_samples(
    samples: np.ndarray,
    observables: Sequence[Observable],
) -> np.ndarray:
    """
    Estimate <Z_S> from bit samples for a fixed list of observables.
    """
    z = bit_samples_to_z_samples(samples)
    num_qubits = z.shape[1]

    corrs = np.empty(len(observables), dtype=float)

    for k, obs in enumerate(observables):
        obs = tuple(obs)

        if len(obs) == 0:
            corrs[k] = 1.0
            continue

        if min(obs) < 0 or max(obs) >= num_qubits:
            raise ValueError(f"Observable {obs} outside num_qubits={num_qubits}.")

        corrs[k] = float(np.mean(np.prod(z[:, obs], axis=1)))

    return corrs


# ---------------------------------------------------------------------------
# IQPopt ideal correlators
# ---------------------------------------------------------------------------

def compute_iqpopt_ideal_corrs_for_observables(
    iqpsimulator: Any,
    iqpsimulator_params: Any,
    observables: Sequence[Observable],
    num_qubits: int,
    n_iqpopt_samples: int = 10000,
    key: Optional[Any] = None,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ideal Pauli-Z correlators with IQPopt for one observable list.

    Parameters
    ----------
    iqpsimulator:
        Object exposing op_expval(params, operators, n_samples, key).
    iqpsimulator_params:
        Parameters passed to op_expval.
    observables:
        List of Pauli-Z supports.
    num_qubits:
        Number of qubits.
    n_iqpopt_samples:
        IQPopt Monte Carlo samples.
    key:
        JAX PRNGKey. If None, uses jax.random.PRNGKey(0).
    batch_size:
        Optional operator batching to reduce memory/runtime spikes.

    Returns
    -------
    expvals:
        Estimated ideal correlators aligned with observables.
    stds:
        IQPopt-reported standard deviations aligned with observables.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    operators = observables_to_operator_matrix(observables, num_qubits, dtype=np.int8)

    if batch_size is None or batch_size >= len(observables):
        expval, std = iqpsimulator.op_expval(
            jnp.array(iqpsimulator_params),
            jnp.array(operators),
            n_iqpopt_samples,
            key,
        )
        return np.asarray(expval, dtype=float), np.asarray(std, dtype=float)

    expval_chunks = []
    std_chunks = []

    for start in range(0, len(observables), batch_size):
        end = min(start + batch_size, len(observables))
        subkey = jax.random.fold_in(key, start)

        expval, std = iqpsimulator.op_expval(
            jnp.array(iqpsimulator_params),
            jnp.array(operators[start:end]),
            n_iqpopt_samples,
            subkey,
        )

        expval_chunks.append(np.asarray(expval, dtype=float))
        std_chunks.append(np.asarray(std, dtype=float))

    return np.concatenate(expval_chunks), np.concatenate(std_chunks)


def compute_iqpopt_ideal_corrs_by_weight(
    iqpsimulator: Any,
    iqpsimulator_params: Any,
    observable_bank: ObservableBank,
    num_qubits: int,
    n_iqpopt_samples: int = 10000,
    key: Optional[Any] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute IQPopt ideal correlators for every weight in an observable bank.

    Returns
    -------
    ideal_corrs:
        ideal_corrs[w] is aligned with observable_bank[w].
    ideal_stds:
        ideal_stds[w] is aligned with observable_bank[w].
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    ideal_corrs: Dict[int, np.ndarray] = {}
    ideal_stds: Dict[int, np.ndarray] = {}

    for weight_idx, (weight, observables) in enumerate(observable_bank.items()):
        weight_key = jax.random.fold_in(key, int(weight_idx))

        expvals, stds = compute_iqpopt_ideal_corrs_for_observables(
            iqpsimulator=iqpsimulator,
            iqpsimulator_params=iqpsimulator_params,
            observables=observables,
            num_qubits=num_qubits,
            n_iqpopt_samples=n_iqpopt_samples,
            key=weight_key,
            batch_size=batch_size,
        )

        ideal_corrs[int(weight)] = expvals
        ideal_stds[int(weight)] = stds

    return ideal_corrs, ideal_stds


def compute_iqpopt_ideal_corrs_for_models(
    iqpopt_objects: Mapping[str, Mapping[str, Any]],
    observable_banks: Mapping[str, ObservableBank],
    num_qubits_by_model: Mapping[str, int],
    n_iqpopt_samples: int = 10000,
    seed: int = 0,
    batch_size: Optional[int] = None,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, np.ndarray]]]:
    """
    Compute IQPopt ideal correlators for all models.

    Parameters
    ----------
    iqpopt_objects:
        iqpopt_objects[model] = {"simulator": simulator, "params": params}
    observable_banks:
        observable_banks[model] = bank
    num_qubits_by_model:
        num_qubits_by_model[model] = n
    n_iqpopt_samples:
        IQPopt samples per op_expval call.
    seed:
        Base seed.
    batch_size:
        Optional batch size for operator evaluation.

    Returns
    -------
    ideal_corrs:
        ideal_corrs[model][weight] = ideal correlator array.
    ideal_stds:
        ideal_stds[model][weight] = IQPopt std array.
    """
    base_key = jax.random.PRNGKey(seed)
    ideal_corrs: Dict[str, Dict[int, np.ndarray]] = {}
    ideal_stds: Dict[str, Dict[int, np.ndarray]] = {}

    for model_idx, model_name in enumerate(iqpopt_objects.keys()):
        obj = iqpopt_objects[model_name]
        model_key = jax.random.fold_in(base_key, model_idx)

        if model_name not in observable_banks:
            raise KeyError(f"Missing observable bank for model {model_name}.")
        if model_name not in num_qubits_by_model:
            raise KeyError(f"Missing num_qubits for model {model_name}.")

        corr_by_w, std_by_w = compute_iqpopt_ideal_corrs_by_weight(
            iqpsimulator=obj["simulator"],
            iqpsimulator_params=obj["params"],
            observable_bank=observable_banks[model_name],
            num_qubits=int(num_qubits_by_model[model_name]),
            n_iqpopt_samples=n_iqpopt_samples,
            key=model_key,
            batch_size=batch_size,
        )

        ideal_corrs[model_name] = corr_by_w
        ideal_stds[model_name] = std_by_w

    return ideal_corrs, ideal_stds


# ---------------------------------------------------------------------------
# Data structure helpers
# ---------------------------------------------------------------------------

def ensure_run_list(x: Union[np.ndarray, Sequence[np.ndarray]]) -> List[np.ndarray]:
    """
    Accept:
      - list/tuple of arrays: [samples_run0, samples_run1, ...]
      - single 2D array: shape (num_samples, num_qubits)
      - 3D array: shape (num_runs, num_samples, num_qubits)

    Return list of 2D arrays.
    """
    if isinstance(x, (list, tuple)):
        return [np.asarray(arr, dtype=np.int8) for arr in x]

    arr = np.asarray(x, dtype=np.int8)

    if arr.ndim == 2:
        return [arr]

    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]

    raise ValueError(f"Unsupported sample container shape: {arr.shape}")


def infer_num_qubits_by_model(samples: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    """
    Infer number of qubits for each model from the first available sample set.
    """
    result: Dict[str, int] = {}

    for model_name, model_samples in samples.items():
        found = False

        for method_samples in model_samples.values():
            runs = ensure_run_list(method_samples)
            if len(runs) > 0:
                result[model_name] = int(runs[0].shape[1])
                found = True
                break

        if not found:
            raise ValueError(f"Could not infer num_qubits for model {model_name}.")

    return result


# ---------------------------------------------------------------------------
# Error table using IQPopt ideal correlators
# ---------------------------------------------------------------------------

def correlator_error_against_ideal_corrs(
    method_samples: np.ndarray,
    ideal_corrs_by_weight: Mapping[int, np.ndarray],
    observable_bank: ObservableBank,
) -> List[Dict[str, Any]]:
    """
    Compare method samples against precomputed ideal correlators.

    ideal_corrs_by_weight[w] must be aligned with observable_bank[w].
    """
    rows: List[Dict[str, Any]] = []

    for weight, observables in observable_bank.items():
        method_corrs = estimate_correlators_from_samples(method_samples, observables)
        ideal_corrs = np.asarray(ideal_corrs_by_weight[int(weight)], dtype=float)

        if len(method_corrs) != len(ideal_corrs):
            raise ValueError(
                f"Weight {weight}: method correlators length {len(method_corrs)} "
                f"does not match ideal correlators length {len(ideal_corrs)}."
            )

        errors = method_corrs - ideal_corrs
        abs_errors = np.abs(errors)

        rows.append({
            "weight": int(weight),
            "num_observables": int(len(observables)),
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "max_abs_error": float(np.max(abs_errors)),
            "mean_signed_error": float(np.mean(errors)),
        })

    return rows


def build_correlator_error_table_from_iqpopt_corrs(
    samples: Mapping[str, Mapping[str, Any]],
    ideal_corrs: Mapping[str, Mapping[int, np.ndarray]],
    observable_banks: Mapping[str, ObservableBank],
    methods: Sequence[str],
) -> pd.DataFrame:
    """
    Build a long-form table using IQPopt ideal correlators.

    Output columns:
    model, method, run, weight, num_observables, mae, rmse,
    max_abs_error, mean_signed_error.
    """
    rows: List[Dict[str, Any]] = []

    for model_name, model_samples in samples.items():
        if model_name not in ideal_corrs:
            raise KeyError(f"Missing ideal_corrs for model {model_name}.")
        if model_name not in observable_banks:
            raise KeyError(f"Missing observable_bank for model {model_name}.")

        bank = observable_banks[model_name]

        for method in methods:
            if method not in model_samples:
                continue

            method_runs = ensure_run_list(model_samples[method])

            for run_idx, method_sample_set in enumerate(method_runs):
                error_rows = correlator_error_against_ideal_corrs(
                    method_samples=method_sample_set,
                    ideal_corrs_by_weight=ideal_corrs[model_name],
                    observable_bank=bank,
                )

                for r in error_rows:
                    r.update({
                        "model": model_name,
                        "method": method,
                        "run": int(run_idx),
                    })
                    rows.append(r)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics and paired comparisons
# ---------------------------------------------------------------------------

def mean_ci95(x: Sequence[float]) -> Tuple[float, float, float]:
    """Mean and normal-approximation 95% CI."""
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(arr))

    if len(arr) == 1:
        return mean, np.nan, np.nan

    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    half_width = 1.96 * se

    return mean, mean - half_width, mean + half_width


def summarize_run_level(df: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """Treat each model-run pair as one unit."""
    rows: List[Dict[str, Any]] = []

    for (method, weight), g in df.groupby(["method", "weight"]):
        mean, lo, hi = mean_ci95(g[metric].values)

        rows.append({
            "method": method,
            "weight": int(weight),
            "metric": metric,
            "num_units": int(len(g)),
            "mean": mean,
            "ci95_low": lo,
            "ci95_high": hi,
        })

    return pd.DataFrame(rows)


def summarize_model_level(df: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """
    First average over runs within each model, then compute CI over models.

    This treats the six circuits/models as the main independent units.
    """
    model_means = (
        df.groupby(["model", "method", "weight"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: f"{metric}_model_mean"})
    )

    rows: List[Dict[str, Any]] = []

    for (method, weight), g in model_means.groupby(["method", "weight"]):
        values = g[f"{metric}_model_mean"].values
        mean, lo, hi = mean_ci95(values)

        rows.append({
            "method": method,
            "weight": int(weight),
            "metric": metric,
            "num_models": int(len(values)),
            "mean": mean,
            "ci95_low": lo,
            "ci95_high": hi,
        })

    return pd.DataFrame(rows)


def paired_method_comparison(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "mae",
    aggregate_over_weights: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Paired comparison of method_a - method_b.

    Negative delta means method_a has lower correlator error than method_b.

    If method_b has one run per model, it is reused across runs of method_a.
    """
    if aggregate_over_weights:
        df_use = df.groupby(["model", "method", "run"], as_index=False)[metric].mean()
        merge_cols = ["model", "run"]
    else:
        df_use = df.copy()
        merge_cols = ["model", "run", "weight"]

    a = df_use[df_use["method"] == method_a].copy()
    b = df_use[df_use["method"] == method_b].copy()

    merged = a.merge(
        b,
        on=merge_cols,
        suffixes=("_a", "_b"),
    )

    if len(merged) == 0:
        # Baseline with one run per model: reuse the baseline for all method_a runs.
        if aggregate_over_weights:
            b_base = b.drop(columns=["run"]).rename(
                columns={metric: f"{metric}_b", "method": "method_b"}
            )
            a_tmp = a.rename(columns={metric: f"{metric}_a", "method": "method_a"})
            merged = a_tmp.merge(b_base, on=["model"])
        else:
            b_base = b.drop(columns=["run"]).rename(
                columns={metric: f"{metric}_b", "method": "method_b"}
            )
            a_tmp = a.rename(columns={metric: f"{metric}_a", "method": "method_a"})
            merged = a_tmp.merge(b_base, on=["model", "weight"])

    if len(merged) == 0:
        raise ValueError("Could not align methods for paired comparison.")

    merged["delta"] = merged[f"{metric}_a"] - merged[f"{metric}_b"]

    rows: List[Dict[str, Any]] = []

    if aggregate_over_weights:
        group_iter = [("all", merged)]
    else:
        group_iter = merged.groupby("weight")

    for weight, g in group_iter:
        mean, lo, hi = mean_ci95(g["delta"].values)

        rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "weight": weight,
            "metric": metric,
            "num_pairs": int(len(g)),
            "mean_delta_a_minus_b": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "fraction_a_better": float(np.mean(g["delta"].values < 0)),
        })

    return pd.DataFrame(rows), merged


def bootstrap_paired_delta(
    paired_df: pd.DataFrame,
    metric_col: str = "delta",
    weight: Union[int, str, None] = None,
    num_bootstrap: int = 5000,
    seed: int = 1234,
) -> Dict[str, float]:
    """
    Cluster bootstrap for paired method differences.

    Resamples models with replacement. Within each selected model, resamples
    rows/runs with replacement.

    Negative delta means method A is better.
    """
    rng = np.random.default_rng(seed)

    if weight is not None and weight != "all":
        data = paired_df[paired_df["weight"] == weight].copy()
    else:
        data = paired_df.copy()

    if len(data) == 0:
        return {
            "mean": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_delta_less_than_0": np.nan,
            "p_delta_greater_than_0": np.nan,
        }

    models = sorted(data["model"].unique())
    boot_means = []

    for _ in range(num_bootstrap):
        selected_models = rng.choice(models, size=len(models), replace=True)
        sampled_values = []

        for model in selected_models:
            g = data[data["model"] == model]
            vals = g[metric_col].values

            if len(vals) == 0:
                continue

            sampled = rng.choice(vals, size=len(vals), replace=True)
            sampled_values.extend(sampled)

        boot_means.append(np.mean(sampled_values))

    boot_means = np.asarray(boot_means, dtype=float)

    return {
        "mean": float(np.mean(data[metric_col].values)),
        "ci95_low": float(np.quantile(boot_means, 0.025)),
        "ci95_high": float(np.quantile(boot_means, 0.975)),
        "p_delta_less_than_0": float(np.mean(boot_means < 0)),
        "p_delta_greater_than_0": float(np.mean(boot_means > 0)),
    }


def bootstrap_paired_comparison_by_weight(
    paired_df: pd.DataFrame,
    num_bootstrap: int = 5000,
    seed: int = 1234,
) -> pd.DataFrame:
    """Bootstrap paired method comparison separately by Pauli-Z weight."""
    rows: List[Dict[str, Any]] = []

    for weight in sorted(paired_df["weight"].unique()):
        result = bootstrap_paired_delta(
            paired_df=paired_df,
            weight=weight,
            num_bootstrap=num_bootstrap,
            seed=seed + int(weight),
        )

        rows.append({
            "weight": int(weight),
            **result,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full IQPopt pipeline
# ---------------------------------------------------------------------------

def run_full_correlator_comparison_with_iqpopt(
    samples: Mapping[str, Mapping[str, Any]],
    iqpopt_objects: Mapping[str, Mapping[str, Any]],
    methods: Sequence[str],
    weights: Sequence[int] = (1, 2, 3, 4),
    num_observables_per_weight: int = 500,
    observable_seed: int = 1000,
    n_iqpopt_samples: int = 10000,
    iqpopt_seed: int = 0,
    batch_size: Optional[int] = None,
    metric: str = "mae",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, ObservableBank],
    Dict[str, Dict[int, np.ndarray]],
    Dict[str, Dict[int, np.ndarray]],
]:
    """
    Full pipeline using IQPopt ideal correlators.

    Parameters
    ----------
    samples:
        samples[model][method][run] -> sample set.
    iqpopt_objects:
        iqpopt_objects[model] = {"simulator": iqpsimulator, "params": params}.
    methods:
        Method names to evaluate.
    weights:
        Pauli-Z weights.
    num_observables_per_weight:
        Random correlators per weight.
    observable_seed:
        Seed for choosing random observables.
    n_iqpopt_samples:
        Number of IQPopt samples for ideal correlator estimation.
    iqpopt_seed:
        Seed for IQPopt estimator.
    batch_size:
        Optional batch size for IQPopt operator calls.
    metric:
        Metric for summary tables, usually "mae" or "rmse".

    Returns
    -------
    df:
        Long-form error table.
    run_summary:
        Summary treating model-run pairs as units.
    model_summary:
        Conservative summary treating models as units.
    observable_banks:
        Random observables used for each model.
    ideal_corrs:
        IQPopt ideal correlators, ideal_corrs[model][weight].
    ideal_stds:
        IQPopt reported standard deviations, ideal_stds[model][weight].
    """
    num_qubits_by_model = infer_num_qubits_by_model(samples)
    model_names = list(samples.keys())

    observable_banks: Dict[str, ObservableBank] = {}
    for model_idx, model_name in enumerate(model_names):
        observable_banks[model_name] = make_observable_bank(
            num_qubits=int(num_qubits_by_model[model_name]),
            weights=weights,
            num_observables_per_weight=num_observables_per_weight,
            seed=observable_seed + model_idx,
        )

    ideal_corrs, ideal_stds = compute_iqpopt_ideal_corrs_for_models(
        iqpopt_objects=iqpopt_objects,
        observable_banks=observable_banks,
        num_qubits_by_model=num_qubits_by_model,
        n_iqpopt_samples=n_iqpopt_samples,
        seed=iqpopt_seed,
        batch_size=batch_size,
    )

    df = build_correlator_error_table_from_iqpopt_corrs(
        samples=samples,
        ideal_corrs=ideal_corrs,
        observable_banks=observable_banks,
        methods=methods,
    )

    run_summary = summarize_run_level(df, metric=metric)
    model_summary = summarize_model_level(df, metric=metric)

    return df, run_summary, model_summary, observable_banks, ideal_corrs, ideal_stds

if __name__ == "__main__":

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
        #samples[run_name]['samples_by_method'] = {sm: [] for sm in methods}
        samples[run_name] = {sm: [] for sm in methods}

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
                samples[run_name]["Product"].append(np.load('./extensive_run_results/{}cap_{}run_{}rep_prod_base_samples.npy'.format(capacity, run_idx, rep)))
                samples[run_name]["DQC+ProdEM_m{}".format(capacity)].append(np.load('./extensive_run_results/{}cap_{}run_{}rep_prod_emsamples.npy'.format(capacity, run_idx, rep)))
                samples[run_name]["DQC+LocalCondEM_m{}".format(capacity)].append(np.load('./extensive_run_results/{}cap_{}run_{}rep_localcond_emsamples.npy'.format(capacity, run_idx, rep)))
                samples[run_name]["DQC_m{}".format(capacity)].append(np.load('./extensive_run_results/{}cap_{}run_{}rep_dqc_samples.npy'.format(capacity, run_idx, rep)))
        iqpopt_objects[run_name] = {"simulator": circuit, "params": params}

    df, run_summary, model_summary, banks, ideal_corrs, ideal_stds = (
            run_full_correlator_comparison_with_iqpopt(
                samples=samples,
                iqpopt_objects=iqpopt_objects,
                methods=methods,
                weights=(1, 2, 3, 4),
                num_observables_per_weight=500,
                observable_seed=1000,
                n_iqpopt_samples=10000,
                iqpopt_seed=0,
                batch_size=250,
                metric="mae",
            )
        )

    df.to_csv('./correlator_comparison.csv')
