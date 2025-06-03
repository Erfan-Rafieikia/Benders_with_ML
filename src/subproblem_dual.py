from gurobipy import Model, GRB, quicksum
import numpy as np
from Synthetic_data_generator import DataGeneral


def _set_params(model: Model) -> None:
    """Set Gurobi solver parameters to suppress output."""
    model.Params.OutputFlag = 0


def solve_dual_subproblem_general(x_val: dict, scenario: int, dat: DataGeneral) -> tuple:
    """
    Solves the dual of the subproblem for a given scenario and fixed first-stage vector x.

    Parameters
    ----------
    x_val : dict
        First-stage decision values: keys are j ∈ J and values are x_j ∈ ℤ.
    scenario : int
        The scenario index m for which the subproblem is solved.
    dat : DataGeneral
        The data instance with all relevant parameters.

    Returns
    -------
    tuple
        A tuple with:
            - dual objective value (float),
            - dual variable vector λ ∈ ℝ_+^{|I|} (as a dictionary).
    """

    with Model("Dual_Subproblem") as mod:
        _set_params(mod)

        # DUAL VARIABLE: λ ∈ ℝ_+^{|I|}
        lam = mod.addVars(dat.I, lb=0, name="lambda")

        # OBJECTIVE: max λᵀ (B x - b_m)
        Bx_minus_b = np.dot(dat.A_matrices[scenario], [x_val[j] for j in dat.J]) - dat.rhs_vectors[scenario]
        obj_expr = quicksum(lam[i] * Bx_minus_b[i] for i in dat.I)
        mod.setObjective(obj_expr, GRB.MAXIMIZE)

        # CONSTRAINT: Aᵀ λ ≤ -ω_m
        A_m = dat.A_matrices[scenario]
        omega_m = dat.omega_matrices[scenario]

        for j in dat.J:
            lhs = quicksum(A_m[i, j] * lam[i] for i in dat.I)
            mod.addConstr(lhs <= -omega_m[j], name=f"DualCon[{j}]")

        # Solve the dual LP
        mod.optimize()

        if mod.status != GRB.OPTIMAL:
            raise RuntimeError(f"Dual subproblem not solved to optimality (status {mod.status}).")

        lambda_vals = mod.getAttr("x", lam)  # Gurobi returns a dictionary
        return mod.ObjVal, lambda_vals
