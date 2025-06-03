"""
p-Median Scenario Selection for Two-Stage Stochastic Optimization

This script implements a facility location-based approach (p-Median) to intelligently sample 
a subset of scenarios for solving subproblems and training ML-based dual variable predictors 
in two-stage stochastic programming.

────────────────────────────────────────────────────────────────────────────
 Overview:
Given a set of feature vectors—one for each scenario—this script formulates and solves 
a p-Median optimization problem. The objective is to select `p` representative scenarios 
(scenario medians) such that the total distance (e.g., Euclidean) between each scenario 
and its assigned representative is minimized.

This sampling method is particularly useful when:
  - Solving subproblems for all scenarios is computationally expensive.
  - You want to reduce the size of the training set for machine learning.
  - You need a balanced and representative training set from a large scenario pool.

────────────────────────────────────────────────────────────────────────────
 Mathematical Formulation:

Minimize:
    ∑_{i∈S} ∑_{j∈S} d_{ij} * y_{ij}

Subject to:
    ∑_{j∈S} x_j = p                         # select exactly p medians
    ∑_{j∈S} y_{ij} = 1 ∀ i∈S                # assign each scenario to one median
    y_{ij} ≤ x_j ∀ i,j∈S                    # only assign to selected medians

Where:
- S is the set of scenarios
- x_j = 1 if scenario j is selected as a median
- y_{ij} = 1 if scenario i is assigned to scenario j
- d_{ij} is the distance between scenarios i and j in the feature space

────────────────────────────────────────────────────────────────────────────
 Inputs:
- feature_vectors: dict[int, np.ndarray]
    Maps each scenario ID to its feature embedding (e.g., from Word2Vec or custom features).
- p: int
    Number of representative scenarios (medians) to select.

────────────────────────────────────────────────────────────────────────────
 Outputs:
- selected_scenarios: list[int]
    The IDs of the selected p representative scenarios.
- assignments: dict[int, int]
    Mapping from each scenario to the scenario it was assigned to (its representative).

────────────────────────────────────────────────────────────────────────────
 Example Usage:
    feature_vectors = generate_scenario_features(data)
    selected_scenarios, _ = select_scenarios(feature_vectors, p=10)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.spatial.distance import cdist


def select_scenarios(feature_vectors, p):
    """
    Selects a representative subset of scenarios (medians) using the p-Median formulation.

    Parameters:
    -----------
    feature_vectors : dict[int, np.ndarray]
        A dictionary mapping each scenario ID to its feature vector.
    p : int
        Number of scenarios to select as representatives (medians).

    Returns:
    --------
    selected_scenarios : list[int]
        The IDs of the p selected representative scenarios.
    assignments : dict[int, int]
        A mapping from each scenario to its assigned representative.
    """
    scenario_ids = list(feature_vectors.keys())
    num_scenarios = len(scenario_ids)

    # Validate input
    if p is None or not isinstance(p, int) or p <= 0:
        raise ValueError(f"Invalid value for p: {p}. It must be a positive integer.")
    if p > num_scenarios:
        raise ValueError(f"p ({p}) cannot be greater than number of scenarios ({num_scenarios}).")

    # Compute pairwise Euclidean distance matrix
    feature_matrix = np.array([feature_vectors[s] for s in scenario_ids])
    distance_matrix = cdist(feature_matrix, feature_matrix, metric="euclidean")

    # Initialize Gurobi model
    model = gp.Model("p-Median")
    model.setParam("OutputFlag", 0)  # Silent mode

    # Binary variables: x[j] = 1 if scenario j is selected
    x = model.addVars(num_scenarios, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_scenarios, num_scenarios, vtype=GRB.BINARY, name="y")  # Assignments

    # Objective: Minimize total distance
    model.setObjective(
        gp.quicksum(distance_matrix[i, j] * y[i, j]
                    for i in range(num_scenarios)
                    for j in range(num_scenarios)),
        GRB.MINIMIZE
    )

    # Constraints
    model.addConstr(gp.quicksum(x[j] for j in range(num_scenarios)) == p, name="Select_p_Medians")

    for i in range(num_scenarios):
        model.addConstr(gp.quicksum(y[i, j] for j in range(num_scenarios)) == 1, name=f"Assign_{i}")

    for i in range(num_scenarios):
        for j in range(num_scenarios):
            model.addConstr(y[i, j] <= x[j], name=f"OnlyAssignToSelected_{i}_{j}")

    # Solve
    model.optimize()

    if model.status == gp.GRB.INFEASIBLE:
        print("❌ Infeasible model. Check input parameters.")
        model.computeIIS()
        model.write("infeasible_model.ilp")
        return [], {}

    if model.status != gp.GRB.OPTIMAL:
        print(f"⚠ Warning: Optimization status = {model.status}")
        return [], {}

    # Extract results
    selected_scenarios = [scenario_ids[j] for j in range(num_scenarios) if x[j].X > 0.5]
    assignments = {
        scenario_ids[i]: scenario_ids[j]
        for i in range(num_scenarios)
        for j in range(num_scenarios)
        if y[i, j].X > 0.5
    }

    return selected_scenarios, assignments
