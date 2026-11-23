import numpy as np
import math

def multi_kernel_hamming_matrix(n_qubits, sigmas):
    """
    Generates a kernel matrix that is the average of several 
    Gaussian kernels with different sigmas.
    """
    dim = 2**n_qubits
    # 1. Generate bitstrings and Hamming distances (the foundation)
    bitstrings = np.array([[int(bit) for bit in format(i, f'0{n_qubits}b')] 
                          for i in range(dim)])
    
    # Efficiently compute pairwise Hamming distances squared
    # Using d^2 is standard for Gaussian kernels
    dists_sq = np.sum(np.square(bitstrings[:, None, :] - bitstrings[None, :, :]), axis=-1)
    
    # 2. Sum the kernels for each sigma
    K_total = np.zeros((dim, dim))
    for s in sigmas:
        # K = exp(-d^2 / (2 * s^2))
        K_total += np.exp(-dists_sq / (2 * s**2))
    
    # Return the average kernel
    return K_total / len(sigmas)

def calculate_multi_mmd(p, q, n_qubits, sigmas=None):
    """
    Calculates MMD^2 using a mixture of kernels.
    Correctly aligns sigma with bit-flip scales.
    """
    if sigmas is None:
        # We want to target scales of k = 1, k = n/4, and k = n/2 bit-flips.
        # To make exp(-k^2 / 2s^2) approx exp(-1), we use s = k / sqrt(2).
        target_k = np.array([1.0, n_qubits/4, n_qubits/2])
        target_k = target_k[target_k > 0] # Avoid div by zero for n=1
        
        # Applying the square root logic for the sigma parameter
        sigmas = target_k / np.sqrt(2)

    K = multi_kernel_hamming_matrix(n_qubits, sigmas)
    
    # MMD^2 = p'Kp + q'Kq - 2p'Kq
    term_pp = p.T @ K @ p
    term_qq = q.T @ K @ q
    term_pq = p.T @ K @ q
    
    return term_pp + term_qq - 2 * term_pq

def hamming_distance_matrix(n_qubits):
    """
    Generates the cost matrix C where C[i,j] is the 
    Hamming distance between bitstring i and j.
    """
    size = 2**n_qubits
    C = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # XOR and count ones gives Hamming distance
            C[i, j] = bin(i ^ j).count('1')
    return C

def bit_samples_to_z_samples(samples):
    samples = np.asarray(samples, dtype=int)
    return 1 - 2 * samples

def estimate_z_correlators(samples, observables):
    """
    Estimate <Z_S> for a fixed list of observables S.
    """
    z_samples = bit_samples_to_z_samples(samples)

    correlators = {}
    for obs in observables:
        obs = tuple(obs)
        if len(obs) == 0:
            correlators[obs] = 1.0
        else:
            correlators[obs] = float(np.mean(np.prod(z_samples[:, obs], axis=1)))

    return correlators

def random_observables(num_qubits, weight, num_observables, seed=None):
    """
    Generate random distinct Pauli-Z supports of fixed weight.
    """
    rng = np.random.default_rng(seed)

    max_possible = math.comb(num_qubits, weight)
    num_observables = min(num_observables, max_possible)

    observables = set()
    while len(observables) < num_observables:
        obs = tuple(sorted(rng.choice(num_qubits, size=weight, replace=False)))
        observables.add(obs)

    return sorted(observables)

def make_random_observable_bank(
    num_qubits,
    max_weight,
    num_observables_per_weight,
    seed=None,
):
    """
    Generate one fixed random observable bank to reuse across methods.

    Returns
    -------
    observable_bank : dict[int, list[tuple[int]]]
        observable_bank[w] is the shared list of weight-w observables.
    """
    rng = np.random.default_rng(seed)
    observable_bank = {}

    for w in range(1, max_weight + 1):
        observable_bank[w] = random_observables(
            num_qubits=num_qubits,
            weight=w,
            num_observables=num_observables_per_weight,
            seed=int(rng.integers(0, 2**32 - 1)),
        )

    return observable_bank

def estimate_correlators_from_bank(samples, observable_bank):
    """
    Estimate correlators for all observables in a fixed observable bank.
    """
    results = {}

    for w, observables in observable_bank.items():
        results[w] = {
            "observables": observables,
            "correlators": estimate_z_correlators(samples, observables),
        }

    return results