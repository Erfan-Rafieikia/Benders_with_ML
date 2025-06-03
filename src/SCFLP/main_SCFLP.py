import os
import numpy as np
import random
from data import read_dataset
from master_problem import solve_problem
from subproblem import solve_subproblem
from sampeled_subproblem import solve_p_median
from feature_generation import generate_feature_vectors
from random_walk import generate_random_walks
from itertools import product
import pandas as pd
from config import *



feature_vectors = generate_feature_vectors(data = data ,n_f = N_F, binary_fraction = BINARY_FRACTION, seed = SEED, n_walk = N_WALK, l_walk = L_WALK, feature_vector_size = FEATURE_VECTOR_SIZE, window = WINDOW_SIZE, min_count = MIN_COUNT, sg = SG)


# Extract representative scenarios using p-median
selected_scenarios, assignments = solve_p_median(feature_vectors, p=P_MEDIAN)

# Solve the 2-stage problem
solution = solve_problem(
        selected_scenarios, feature_vectors, data,
        prediction_method="knn", n_neighbors=5, use_prediction=USE_PREDICTION
)




# Define parameter grid
data_files = ["p1"]  # List of dataset file names
use_prediction_options = [True, False]
variance_factors = [0.5,1.5]
p_median_values = [5,10]
NUM_SCENARIOS=60
N_F = 20

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "../data/")
output_file = os.path.join(script_dir, "stochastic_cflp_results.xlsx")

# DataFrame to store results
results = []

for DATA_FILE, USE_PREDICTION, VARIANCE_FACTOR, P_MEDIAN_VALUE in product(
    data_files, use_prediction_options, variance_factors, p_median_values
):
    
    datafile = os.path.join(DATA_DIR, DATA_FILE)
    data = read_dataset(datafile, num_scenarios=NUM_SCENARIOS, variance_factor=VARIANCE_FACTOR)

    # Generate y values
    y_values_features = generate_y_values(n_f=N_F, data=data, binary=False, seed=SEED)
    Dual_values_features = calculate_dual(y_values_features, data)
    weight_values_features = calculate_weight(Dual_values_features, y_values_features)

    # Generate scenario features
    C = generate_random_walks(weight_values_features, n_walk=10, l_walk=20, seed=SEED)
    model = learn_subproblem_features(C, w=5)
    feature_vectors = {scenario: model.wv[scenario] for scenario in model.wv.index_to_key}

    # Solve p-median problem
    selected_scenarios, assignments = solve_p_median(feature_vectors, p=P_MEDIAN_VALUE)

    # Solve the stochastic CFLP
    solution = solve_problem(
        selected_scenarios, feature_vectors, data,
        prediction_method="knn", n_neighbors=5, use_prediction=USE_PREDICTION
    )

    # Store results
    results.append({
        "DATA_FILE": DATA_FILE,
        "USE_PREDICTION": USE_PREDICTION,
        "VARIANCE_FACTOR": VARIANCE_FACTOR,
        "P_MEDIAN_VALUE": P_MEDIAN_VALUE,
        "Objective Value": solution.objective_value,
        "Open Facilities": [j for j in data.J if solution.locations[j] > 0.5],
        "Solution Time (sec)": solution.solution_time,
        "BD Cuts (MIP) Selected": sum(solution.num_cuts_mip_selected.values()) if solution.num_cuts_mip_selected else 0,
        "BD Cuts (Relaxed) Selected": sum(solution.num_cuts_rel_selected.values()) if solution.num_cuts_rel_selected else 0,
        "BD Cuts (MIP) ML": sum(solution.num_cuts_mip_ml.values()) if solution.num_cuts_mip_ml else 0,
        "BD Cuts (Relaxed) ML": sum(solution.num_cuts_rel_ml.values()) if solution.num_cuts_rel_ml else 0,
        "BD Cuts (MIP) Unselected": sum(solution.num_cuts_mip_unselected.values()) if solution.num_cuts_mip_unselected else 0,
        "BD Cuts (Relaxed) Unselected": sum(solution.num_cuts_rel_unselected.values()) if solution.num_cuts_rel_unselected else 0,
        "Explored Branch-and-Bound Nodes": solution.num_bnb_nodes,
    })

# Convert results to a DataFrame and save to Excel
df_results = pd.DataFrame(results)
df_results.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")