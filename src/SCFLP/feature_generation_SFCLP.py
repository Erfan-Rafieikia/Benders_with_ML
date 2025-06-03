import os
import numpy as np
import random
import pandas as pd
from itertools import product
from gensim.models import Word2Vec

from config import *
from data import read_dataset
from subproblem_SFCLP import solve_subproblem
from random_walk_SFCLP import generate_random_walks

def learn_subproblem_features(C, feature_vector_size, window, min_count, sg):
    """
    Learn subproblem features using Word2Vec.
    """
    # Train Word2Vec model on random walks
    model = Word2Vec(
        sentences=C,
        vector_size=feature_vector_size,
        window=window,
        min_count=min_count,
        sg=sg
    )
    return model

def generate_first_stage_trial(n_f, data, binary_fraction, seed):
    # Set random seed for reproducibility
    np.random.seed(seed)
    # Initialize array for first-stage trials
    first_stage_trials = np.zeros((n_f, len(data.J)))
    num_binary = int(len(data.J) * binary_fraction)  # Number of binary variables
    binary_indices = np.arange(len(data.J))
    for i in range(n_f):
        valid = False
        while not valid:
            np.random.shuffle(binary_indices)  # Shuffle indices for random selection
            bin_idx = binary_indices[:num_binary]  # Indices for binary variables
            cont_idx = binary_indices[num_binary:]  # Indices for continuous variables
            y_row = np.zeros(len(data.J))
            # Assign random binary values
            y_row[bin_idx] = np.random.choice([0, 1], size=num_binary)
            # Assign random continuous values
            y_row[cont_idx] = np.random.uniform(0, 1, size=len(data.J) - num_binary)
            # Check capacity constraint
            if sum(data.capacities[j] * y_row[j] for j in data.J) >= data.max_demand_sum_over_scenario:
                valid = True
                first_stage_trials[i] = y_row
    return first_stage_trials

def calculate_dual(first_stage_trials, data):
    first_stage_decision_count = first_stage_trials.shape[0]
    dual_values = {}
    for count in range(first_stage_decision_count):
        for s in data.S:
            # Solve subproblem for each scenario and trial
            obj, mu, nu = solve_subproblem(data, first_stage_trials[count], s)
            dual_values[(count, s)] = (mu, nu)
    return dual_values

def distance_function(mu_nu_si, mu_nu_sj):
    # Unpack dual variables
    mu_si, nu_si = mu_nu_si
    mu_sj, nu_sj = mu_nu_sj
    # Sort keys for consistent ordering
    keys_mu = sorted(mu_si.keys())
    keys_nu = sorted(nu_si.keys())
    # Convert duals to vectors
    mu_si_values = np.array([mu_si[key] for key in keys_mu])
    mu_sj_values = np.array([mu_sj[key] for key in keys_mu])
    nu_si_values = np.array([nu_si[key] for key in keys_nu])
    nu_sj_values = np.array([nu_sj[key] for key in keys_nu])
    # Concatenate dual vectors
    vec_si = np.concatenate([mu_si_values, nu_si_values])
    vec_sj = np.concatenate([mu_sj_values, nu_sj_values])
    # Compute Euclidean distance
    distance = np.linalg.norm(vec_si - vec_sj)
    # Normalize distance
    norm_factor = np.linalg.norm(vec_si) + np.linalg.norm(vec_sj)
    normalized_distance = distance / norm_factor if norm_factor > 0 else 0
    return normalized_distance

def calculate_weight(dual_values, y_values):
    first_stage_decision_count = y_values.shape[0]
    scenarios = set(s for _, s in dual_values.keys())
    weight_values = {}
    for s_i in scenarios:
        for s_j in scenarios:
            if s_i != s_j:
                total_difference = 0
                for count in range(1, first_stage_decision_count + 1):
                    # Get duals for both scenarios
                    mu_nu_si = dual_values.get((count, s_i))
                    mu_nu_sj = dual_values.get((count, s_j))
                    if mu_nu_si is not None and mu_nu_sj is not None:
                        # Accumulate distance between duals
                        total_difference += distance_function(mu_nu_si, mu_nu_sj)
                # Average distance over all trials
                weight_values[(s_i, s_j)] = total_difference / first_stage_decision_count
    return weight_values

def generate_feature_vectors(data,n_f,binary_fraction,seed,n_walk,l_walk,feature_vector_size,window,min_count,sg):
    """
    Generates feature vectors for scenarios using dual variable sensitivity and Word2Vec.
    """
    # Generate random first-stage decisions
    first_stage_trials = generate_first_stage_trial(n_f, data, binary_fraction, seed)
    # Compute dual values for each trial and scenario
    dual_values_trials = calculate_dual(first_stage_trials, data)
    # Calculate scenario-to-scenario weights based on dual sensitivity
    weight_values_features = calculate_weight(dual_values_trials, first_stage_trials)
    # Generate random walks over scenario graph
    C = generate_random_walks(weight_values_features, n_walk=n_walk, l_walk=l_walk, seed=seed)
    # Learn scenario embeddings using Word2Vec
    model = learn_subproblem_features(
        C,
        feature_vector_size=feature_vector_size,
        window=window,
        min_count=min_count,
        sg=sg
    )
    # Extract feature vectors for each scenario
    feature_vectors = {scenario: model.wv[scenario] for scenario in model.wv.index_to_key}
    return feature_vectors
