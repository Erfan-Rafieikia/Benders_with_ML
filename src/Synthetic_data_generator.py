import numpy as np
from dataclasses import dataclass


@dataclass
class DataGeneral:
    """
    Stores synthetic data for a two-stage stochastic optimization problem.
    Includes scenario-dependent and deterministic inputs.
    """
    I: np.ndarray                          # Index set for second-stage variables (|I|)
    J: np.ndarray                          # Index set for first-stage variables (|J|)
    M: np.ndarray                          # Index set for scenarios (|M| = num_scenarios)
    rhs_vectors: np.ndarray                # Scenario RHS vectors b_m ∈ ℝ^{|I|}, shape: (|M| × |I|)
    A_matrices: list                       # List of scenario matrices A_m ∈ ℤ^{|I| × |J|}, length: |M|
    omega_matrices: list                   # List of scenario cost vectors ω_m ∈ ℝ^{|J|}, length: |M|
    cost_vector_x: np.ndarray              # First-stage cost vector c ∈ ℝ^{|J|}
    aux_cost_matrix: np.ndarray            # Auxiliary cost matrix (optional), shape: (|I| × |J|)
    D_matrix: np.ndarray                   # Global constraint matrix D ∈ ℝ^{|K| × |J|}
    d_vector: np.ndarray                   # RHS vector for D x ≤ d ∈ ℝ^{|K|}
    num_scenarios: int                     # Number of scenarios (|M|)
    scenario_dist_type: dict               # Distribution type used per parameter
    scenario_variance: dict                # Variance factor used per parameter
    value_ranges: dict                     # Ranges used for synthetic data generation

def generate_random_instance(
    size_I,
    size_J,
    num_scenarios,
    num_x_constraints=3,
    base_rhs_range=(5, 15),
    A_base_range=(0, 4),
    omega_base_range=(0, 10),
    c_range=(0, 20),
    aux_range=(1, 10),
    D_range=(1, 5),
    d_range=(10, 30),
    rhs_variance_factor=0.2,
    A_variance_factor=0.1,
    omega_variance_factor=0.15,
    rhs_method="normal",
    A_method="normal",
    omega_method="normal",
    shared_matrix=False,
    shared_objective=False,
    seed=0
) -> DataGeneral:
    """
    Generate synthetic data for a two-stage stochastic MILP:
        min_x cᵀ x + Σ_m ω_mᵀ y^m
        s.t.   A_m y^m + B_m x ≤ b_m  ∀m
               D x ≤ d
               x ∈ ℤ^{|J|}, y^m ∈ ℝ_+^{|I|}

    Returns
    -------
    DataGeneral
        Structured data object containing all model components.
    """
    np.random.seed(seed)

    I = np.arange(size_I)
    J = np.arange(size_J)
    M = np.arange(num_scenarios)

    base_rhs = np.random.randint(base_rhs_range[0], base_rhs_range[1] + 1, size=size_I)

    def generate_noise(base, variance, method, is_integer=False):
        results = []
        for _ in range(num_scenarios):
            if method == "normal":
                noisy = np.random.normal(loc=base, scale=variance * base)
            elif method == "uniform":
                noisy = np.random.uniform(low=base - variance * base, high=base + variance * base)
            elif method == "poisson":
                noisy = np.random.poisson(lam=base)
            elif method == "lognormal":
                noisy = np.random.lognormal(mean=np.log(base + 1), sigma=variance)
            else:
                raise NotImplementedError(f"Method '{method}' not supported.")
            noisy = np.maximum(noisy, 0)
            if is_integer:
                noisy = np.round(noisy).astype(int)
            results.append(noisy)
        return results

    rhs_vectors = generate_noise(base_rhs, rhs_variance_factor, rhs_method, is_integer=True)

    base_A = np.random.randint(A_base_range[0], A_base_range[1] + 1, size=(size_I, size_J))
    A_matrices = [base_A for _ in range(num_scenarios)] if shared_matrix else \
                 generate_noise(base_A, A_variance_factor, A_method, is_integer=True)

    base_omega = np.random.uniform(omega_base_range[0], omega_base_range[1], size=size_J)
    omega_matrices = [base_omega for _ in range(num_scenarios)] if shared_objective else \
                     generate_noise(base_omega, omega_variance_factor, omega_method, is_integer=False)

    cost_vector_x = np.random.uniform(c_range[0], c_range[1], size=size_J).round(2)
    aux_cost_matrix = np.random.uniform(aux_range[0], aux_range[1], size=(size_I, size_J)).round(2)

    D_matrix = np.random.randint(D_range[0], D_range[1] + 1, size=(num_x_constraints, size_J))
    d_vector = np.random.randint(d_range[0], d_range[1] + 1, size=num_x_constraints)

    return DataGeneral(
        I=I,
        J=J,
        M=M,
        rhs_vectors=np.array(rhs_vectors),
        A_matrices=A_matrices,
        omega_matrices=omega_matrices,
        cost_vector_x=cost_vector_x,
        aux_cost_matrix=aux_cost_matrix,
        D_matrix=D_matrix,
        d_vector=d_vector,
        num_scenarios=num_scenarios,
        scenario_dist_type={
            "rhs": rhs_method,
            "A": A_method,
            "omega": omega_method
        },
        scenario_variance={
            "rhs": rhs_variance_factor,
            "A": A_variance_factor,
            "omega": omega_variance_factor
        },
        value_ranges={
            "base_rhs_range": base_rhs_range,
            "A_base_range": A_base_range,
            "omega_base_range": omega_base_range,
            "c_range": c_range,
            "aux_range": aux_range,
            "D_range": D_range,
            "d_range": d_range
        }
    )
