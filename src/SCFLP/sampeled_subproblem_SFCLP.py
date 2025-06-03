import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.spatial.distance import cdist

def solve_p_median(feature_vectors, p):
    """
    Solves the p-Median problem using Gurobi.

    Parameters:
        feature_vectors (dict): A dictionary where keys are scenario IDs and values are feature vectors.
        p (int): Number of sampled scenarios (medians).

    Returns:
        selected_scenarios (list): List of sampled scenario IDs (medians).
        assignments (dict): Dictionary mapping each scenario to its assigned median scenario.
    """
    scenario_ids = list(feature_vectors.keys())  # Scenario indices
    num_scenarios = len(scenario_ids)

    # Ensure p is valid
    if p is None or not isinstance(p, int) or p <= 0:
        raise ValueError(f"Invalid value for p: {p}. It must be a positive integer.")
    if p > num_scenarios:
        raise ValueError(f"p ({p}) cannot be greater than the number of scenarios ({num_scenarios}).")

    # Convert feature vectors into a matrix
    feature_matrix = np.array([feature_vectors[s] for s in scenario_ids])

    # Compute pairwise Euclidean distances
    distance_matrix = cdist(feature_matrix, feature_matrix, metric="euclidean")

    # Create Gurobi model
    model = gp.Model("p-Median")

    # Decision Variables
    x = model.addVars(num_scenarios, vtype=GRB.BINARY, name="x")  # 1 if scenario is a median
    y = model.addVars(num_scenarios, num_scenarios, vtype=GRB.BINARY, name="y")  # 1 if assigned to a median

    # Objective: Minimize total assignment distance
    model.setObjective(
        gp.quicksum(distance_matrix[i, j] * y[i, j] for i in range(num_scenarios) for j in range(num_scenarios)),
        GRB.MINIMIZE
    )

    # Constraint: Select exactly `p` sampled scenarios
    model.addConstr(gp.quicksum(x[i] for i in range(num_scenarios)) == p, name="Select_p_Medians")

    # Constraint: Each scenario is assigned to exactly one sampled scenario
    for i in range(num_scenarios):
        model.addConstr(gp.quicksum(y[i, j] for j in range(num_scenarios)) == 1, name=f"Assign_{i}")

    # Constraint: A scenario can only be assigned to a selected median
    for i in range(num_scenarios):
        for j in range(num_scenarios):
            model.addConstr(y[i, j] <= x[j], name=f"AssignOnlyIfSelected_{i}_{j}")

    # Solve the model
    model.optimize()

    # Handle infeasibility
    if model.status == gp.GRB.INFEASIBLE:
        print("❌ Model is infeasible. Running infeasibility analysis...")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Save constraints causing infeasibility
        return [], {}

    # Handle cases where no optimal solution is found
    if model.status != gp.GRB.OPTIMAL:
        print(f"⚠ Warning: Model did not find an optimal solution. Status: {model.status}")
        return [], {}

    # Extract selected medians
    selected_scenarios = [scenario_ids[i] for i in range(num_scenarios) if x[i].X > 0.5]

    # Extract assignments (mapping each scenario to its closest sampled scenario)
    assignments = {scenario_ids[i]: scenario_ids[j]
                   for i in range(num_scenarios)
                   for j in range(num_scenarios) if y[i, j].X > 0.5}

    return selected_scenarios, assignments