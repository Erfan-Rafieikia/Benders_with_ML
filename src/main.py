"""
main.py

This script executes experimental evaluations of a two-stage stochastic optimization problem using different solution strategies:
- Classic Benders decomposition
- Regression-based ML dual prediction
- K-Nearest Neighbors (KNN) dual prediction

For each problem instance configuration, it:
1. Generates synthetic problem data.
2. Generates feature vectors for all scenarios.
3. Selects a subset of representative scenarios using p-median clustering.
4. Solves the master problem with different prediction methods.
5. Records detailed results including objective value, timing, and number of cuts.

Results are stored in a CSV file named "experiment_results.csv".
"""

import time
import pandas as pd

from feature_generation import generate_scenario_features
from scenario_selection import select_scenarios
from Synthetic_data_generator import generate_random_instance as generate_data_general
from master_problem import solve_master_problem_general as solve_master_problem

# Settings for each problem instance
problem_instances = [
    {"num_x": 10, "num_y": 5, "num_scenarios": 50, "seed": 0},
    {"num_x": 20, "num_y": 10, "num_scenarios": 100, "seed": 1},
    {"num_x": 30, "num_y": 15, "num_scenarios": 150, "seed": 2},
]

# Output records
records = []

for setting in problem_instances:
    print("\nSolving problem instance:", setting)

    # 1. Data generation
    t0 = time.time()
    data = generate_data_general(
        num_x=setting["num_x"],
        num_y=setting["num_y"],
        num_scenarios=setting["num_scenarios"],
        seed=setting["seed"]
    )
    time_data = time.time() - t0

    # 2. Feature generation
    t0 = time.time()
    feature_vectors = generate_scenario_features(data)
    time_feature = time.time() - t0

    # 3. Scenario selection (p-median)
    t0 = time.time()
    selected_scenarios, _ = select_scenarios(feature_vectors, p=10)
    time_selection = time.time() - t0

    # 4. Solve for each method
    for method in ["classic", "regression", "knn"]:
        print(f"\nSolving with method: {method}")
        t0 = time.time()

        results = solve_master_problem(
            dat=data,
            feature_vectors=feature_vectors,
            selected_scenarios=selected_scenarios,
            prediction_method=method,
            use_prediction=(method != "classic"),
            allow_fractional_x=True,
            n_neighbors=5  # For KNN
        )
        time_solve = results.solution_time
        total_cuts = lambda cut_dict: sum(cut_dict.values()) if isinstance(cut_dict, dict) else 0

        # Save row
        records.append({
            "num_x": setting["num_x"],
            "num_y": setting["num_y"],
            "num_scenarios": setting["num_scenarios"],
            "seed": setting["seed"],
            "method": method,
            "obj_val": results.objective_value,
            "time_data": time_data,
            "time_feature": time_feature,
            "time_selection": time_selection,
            "time_solve": time_solve,
            "cuts_mip_selected": total_cuts(results.num_cuts_mip_selected),
            "cuts_rel_selected": total_cuts(results.num_cuts_rel_selected),
            "cuts_mip_ml": total_cuts(results.num_cuts_mip_ml),
            "cuts_rel_ml": total_cuts(results.num_cuts_rel_ml),
            "cuts_mip_unselected": total_cuts(results.num_cuts_mip_unselected),
            "cuts_rel_unselected": total_cuts(results.num_cuts_rel_unselected),
            "training_success_rate": getattr(results, "training_success_rate", 0.0)
        })

# Save to CSV
pd.DataFrame(records).to_csv("experiment_results.csv", index=False)
print("\nAll experiments completed. Results saved to experiment_results.csv")
