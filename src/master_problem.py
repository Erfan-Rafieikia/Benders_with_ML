from dataclasses import dataclass
from gurobipy import Model, GRB, quicksum
from callbacks import Callback
from Synthetic_data_generator import DataGeneral
from config_SCFLP import *  # Optional config file with default ML settings

@dataclass
class Solution:
    """
    Stores the solution to the master problem along with metadata and cut statistics.
    """
    objective_value: float
    x_values: dict
    solution_time: float
    num_cuts_mip_selected: dict
    num_cuts_rel_selected: dict
    num_cuts_mip_ml: dict
    num_cuts_rel_ml: dict
    num_cuts_mip_unselected: dict
    num_cuts_rel_unselected: dict
    num_bnb_nodes: int = 0

def _set_params(mod: Model):
    """
    Sets Gurobi parameters for solving the master problem.
    """
    mod.Params.LazyConstraints = 1
    # mod.Params.TimeLimit = 600  # Optional: set a time limit

def solve_master_problem_general(
    selected_scenarios,
    feature_vectors,
    dat: DataGeneral,
    prediction_method=PREDICTION_METHOD,
    n_neighbors=N_NEIGHBORS,
    use_prediction=USE_PREDICTION,
    allow_fractional_x=True,
    write_lp=False
):
    """
    Solves the master problem for a general two-stage stochastic program with optional ML-enhanced callback.

    Parameters
    ----------
    selected_scenarios : list[int]
        Indices of scenarios for which cuts are generated directly.
    feature_vectors : dict[int, np.ndarray]
        Feature vectors for ML-based prediction.
    dat : DataGeneral
        Problem data object with matrices and vectors.
    prediction_method : str
        Type of predictor to use ('regression', 'knn', etc.).
    n_neighbors : int
        Number of neighbors for KNN.
    use_prediction : bool
        Whether to use ML for unselected scenarios.
    allow_fractional_x : bool
        If True, allow fractional values for x in callbacks.
    write_lp : bool
        If True, exports the LP file.

    Returns
    -------
    Solution
        A dataclass holding the optimal objective value, solution time, x values, and cut statistics.
    """
    with Model("General_Master") as mod:
        _set_params(mod)

        # Variables
        x = mod.addVars(dat.J, vtype=GRB.INTEGER, name="x")
        eta = mod.addVars(dat.M, vtype=GRB.CONTINUOUS, name="eta")

        # Objective: cᵀx + (1/|M|) Σ η_m
        total_cost = quicksum(dat.cost_vector_x[j] * x[j] for j in dat.J) + \
                     quicksum(eta[m] for m in dat.M) / len(dat.M)
        mod.setObjective(total_cost, GRB.MINIMIZE)

        # Global constraint: D x ≤ d
        for k in range(dat.D_matrix.shape[0]):
            mod.addConstr(
                quicksum(dat.D_matrix[k, j] * x[j] for j in dat.J) <= dat.d_vector[k],
                name=f"GlobalConstraint[{k}]"
            )

        # Callback setup
        callback = Callback(
            dat=dat,
            x_vars=x,
            eta_vars=eta,
            selected_scenarios=selected_scenarios,
            feature_vectors=feature_vectors,
            prediction_method=prediction_method,
            n_neighbors=n_neighbors,
            use_prediction=use_prediction,
            allow_fractional_x=allow_fractional_x
        )

        if write_lp:
            mod.write(f"{mod.ModelName}.lp")

        # Solve
        mod.optimize(callback)

        # Extract solution
        x_sol = mod.getAttr("x", x)

        return Solution(
            objective_value=mod.ObjVal,
            x_values=x_sol,
            solution_time=mod.Runtime,
            num_cuts_mip_selected=callback.num_cuts_mip_selected,
            num_cuts_rel_selected=callback.num_cuts_rel_selected,
            num_cuts_mip_ml=callback.num_cuts_mip_ml,
            num_cuts_rel_ml=callback.num_cuts_rel_ml,
            num_cuts_mip_unselected=callback.num_cuts_mip_unselected,
            num_cuts_rel_unselected=callback.num_cuts_rel_unselected,
            num_bnb_nodes=mod.NodeCount
        )
