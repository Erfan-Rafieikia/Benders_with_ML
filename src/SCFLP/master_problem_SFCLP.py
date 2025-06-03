from dataclasses import dataclass
from config import *
from callbacks import Callback
from data import Data
from gurobipy import GRB, Model, quicksum

@dataclass
@dataclass
class Solution:
    objective_value: float
    locations: list
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
    Set the parameters of the Gurobi solver.

    Args:
        mod (Model): The Gurobi model for which the parameters are being set.
    """

    # Enable lazy constraint adaptation for optimality cuts
    mod.Params.LazyConstraints = 1

    # Use the following to set a time limit for the solver
    # mod.Params.TimeLimit = 60.0


def solve_problem(selected_scenarios, feature_vectors, dat, prediction_method=PREDICTION_METHOD, n_neighbors=N_NEIGHBORS, use_prediction=USE_PREDICTION):
    use_prediction = use_prediction

    with Model("FLP_Master") as mod:
        _set_params(mod)

        y = mod.addVars(dat.J, vtype=GRB.BINARY, name="y")
        eta = mod.addVars(dat.S, name="eta")

        total_cost = quicksum(dat.fixed_costs[j] * y[j] for j in dat.J) +  \
                     quicksum(eta[s] for s in dat.S) / len(dat.S)
        mod.setObjective(total_cost, sense=GRB.MINIMIZE)

        mod.addConstr(
            quicksum(dat.capacities[j] * y[j] for j in dat.J) >= dat.max_demand_sum_over_scenario,
            name="Feasibility"
        )

        callback = Callback(dat, y, eta, selected_scenarios, feature_vectors, prediction_method=prediction_method, n_neighbors=n_neighbors, use_prediction=use_prediction)

        write_mp_lp = False  # Ensure this is defined
        if write_mp_lp:
            mod.write(f"{mod.ModelName}.lp")

        mod.optimize(callback)

        obj = mod.ObjVal
        sol_time = round(mod.Runtime, 2)
        y_values = mod.getAttr("x", y)

        return Solution(
            obj,
            y_values,
            sol_time,
            callback.num_cuts_mip_selected,
            callback.num_cuts_rel_selected,
            callback.num_cuts_mip_ml,
            callback.num_cuts_rel_ml,
            callback.num_cuts_mip_unselected,
            callback.num_cuts_rel_unselected,
            int(mod.NodeCount)
        )