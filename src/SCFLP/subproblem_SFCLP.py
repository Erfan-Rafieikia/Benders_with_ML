from data import Data
from gurobipy import GRB, Model, quicksum, tupledict


def _set_params(model: Model) -> None:
    """Set the parameters for the Gurobi solver to suppress console output."""
    model.Params.OutputFlag = 0


def solve_subproblem(dat: Data, facility_open: tupledict, scenario) -> tuple:
    """
    Solve the subproblem for the Facility Location Problem (FLP).

    This function defines and optimizes the subproblem given the facility open decisions.
    It calculates the optimal shipment quantities from facilities to customers minimizing
    the total shipment costs while satisfying demand and capacity constraints.

    Args:
        dat (Data): The input data containing costs, demands, and capacities.
        facility_open (tupledict): A dictionary with facility indices as keys and binary
                                   decisions (1 if open, 0 if closed) as values.

    Returns:
        tuple: A tuple containing the objective value, dual values for demand constraints
               (mu), and dual values for capacity constraints (nu).
    """

    with Model("FLP_Sub") as mod:
        _set_params(mod)

        # Decision variables for shipment quantities from facilities to customers
        ##My modificatin for assignment is making x a binary variable.
        x = mod.addVars(dat.I, dat.J,dat.S, name="x")

        # Objective: Minimize total shipment costs
        total_cost = quicksum(dat.shipment_costs[i, j] * x[i, j, scenario] for i in dat.I for j in dat.J)
        mod.setObjective(total_cost, GRB.MINIMIZE)

        # Constraints: Satisfy demand for each customer
        demand_constraints = mod.addConstrs(
            (quicksum(x[i, j, scenario] for j in dat.J) >= dat.scenarios[scenario, i] for i in dat.I),
            name="Demand",
        )

        # Constraints: Do not exceed capacity for open facilities
        capacity_constraints = mod.addConstrs(
            (
                quicksum(x[i, j, scenario] for i in dat.I) <= dat.capacities[j] * facility_open[j]
                for j in dat.J
            ),
            name="Capacity",
        )

        # Optimize the model to find the best shipment plan
        mod.optimize()

        # Retrieve the objective value and dual values from optimized model
        objective_value = mod.ObjVal
        mu_values = mod.getAttr("pi", demand_constraints)
        nu_values = mod.getAttr("pi", capacity_constraints)

    return objective_value, mu_values, nu_values