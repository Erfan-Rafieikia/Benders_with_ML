"""
Feature Generation for Stochastic Programs via Dual Sensitivity and Random Walk Embeddings
-------------------------------------------------------------------------------------------

This module generates feature vectors for each scenario in a two-stage stochastic program
by analyzing **dual sensitivity** across first-stage decisions and constructing scenario
embeddings using **Word2Vec** on biased random walks over a scenario similarity graph.

The steps are as follows:

1. **Generate Valid First-Stage Decisions**:
   A set of `n_trials` feasible first-stage solutions (vectors `x`) is randomly created.
   Each vector satisfies the linear constraint:      D * x ≤ d
   Some components of `x` are binary (0 or 1), while others are continuous in [0, 1],
   controlled by the `binary_fraction` parameter.

2. **Solve Dual Subproblems**:
   For each x and each scenario m ∈ M, the dual subproblem is solved.
   The dual vectors λₘ(x) are stored for measuring scenario sensitivity.

3. **Measure Scenario Distances**:
   A similarity graph is built by computing pairwise distances between scenarios m₁ and m₂.
   For each pair (m₁, m₂), the average distance across all x is computed as:

       d(m₁, m₂) = (1 / T) * Σ_{t=1}^T  [‖λₘ₁(xₜ) - λₘ₂(xₜ)‖ / (‖λₘ₁(xₜ)‖ + ‖λₘ₂(xₜ)‖)]

   This is a normalized Euclidean distance that reflects how similarly the scenarios react
   to changes in first-stage decisions.

4. **Random Walks over the Scenario Graph**:
   Random walks are simulated over the scenario graph (nodes = scenarios, edges = similarity)
   to capture contextual relationships among scenarios. Transition probabilities are biased
   by scenario similarity (closer = higher chance of walking there).

5. **Train Word2Vec Embedding**:
   The random walks are treated as "sentences" and scenarios as "words". Word2Vec learns
   low-dimensional dense vector representations (embeddings) for each scenario that capture
   structural similarity in the graph.

Parameters and Meanings:
------------------------
- `n_trials`: Number of x samples generated
- `binary_fraction`: Fraction of x components that are binary (rest are in [0,1])
- `seed`: Random seed for reproducibility
- `n_walks`: Number of random walks started per scenario
- `walk_length`: Number of steps in each random walk
- `feature_dim`: Size of the scenario embedding vector (i.e., dimension of output)
- `window`: Word2Vec context window (how many neighbors are seen during training)
- `min_count`: Minimum number of times a scenario ID must appear to be included
- `sg`: Word2Vec training algorithm:
    * `sg=0`: CBOW (Continuous Bag-of-Words)
    * `sg=1`: Skip-Gram (preferred for sparse data and better for rare tokens)

Returns:
--------
- `feature_vectors`: A dictionary mapping scenario IDs (int) to NumPy arrays (embeddings)

Usage Example:
--------------
```python
features = generate_scenario_features(
    data=my_data_object,
    n_trials=50,
    binary_fraction=0.4,
    seed=123,
    n_walks=25,
    walk_length=15,
    feature_dim=10,
    window=5,
    min_count=1,
    sg=1
)
"""

import numpy as np
from itertools import product
from gensim.models import Word2Vec
from subproblem_dual import solve_dual_subproblem_general

def generate_first_stage_trials(n_trials, data, binary_fraction, seed):
    np.random.seed(seed)
    x_trials = []
    n_binary = int(binary_fraction * len(data.J))
    indices = list(range(len(data.J)))

    while len(x_trials) < n_trials:
        np.random.shuffle(indices)
        bin_idx = indices[:n_binary]
        cont_idx = indices[n_binary:]

        x = np.zeros(len(data.J))
        x[bin_idx] = np.random.choice([0, 1], size=n_binary)
        x[cont_idx] = np.random.uniform(0, 1, size=len(cont_idx))

        if np.all(data.D_matrix @ x <= data.d_vector + 1e-6):
            x_trials.append(x)

    return np.array(x_trials)

def get_dual_vectors(x_trials, data):
    duals = {}
    for i, x in enumerate(x_trials):
        for m in data.M:
            _, lambda_vec = solve_dual_subproblem_general({j: x[j] for j in data.J}, m, data)
            # Convert to NumPy array for compatibility
            duals[(i, m)] = np.array([lambda_vec[i_] for i_ in data.I])
    return duals

def dual_distance(dual_1, dual_2):
    norm = np.linalg.norm(dual_1) + np.linalg.norm(dual_2)
    return np.linalg.norm(dual_1 - dual_2) / norm if norm > 0 else 0

def compute_scenario_weights(duals, x_trials, M):
    weights = {}
    for m1, m2 in product(M, M):
        if m1 == m2:
            continue
        total = 0
        for i in range(len(x_trials)):
            d1 = duals.get((i, m1))
            d2 = duals.get((i, m2))
            if d1 is not None and d2 is not None:
                total += dual_distance(d1, d2)
        weights[(m1, m2)] = total / len(x_trials)
    return weights

def generate_random_walks(weights, M, n_walks, walk_length, seed):
    np.random.seed(seed)
    walks = []
    for _ in range(n_walks):
        for start in M:
            walk = [start]
            current = start
            for _ in range(walk_length):
                neighbors = [t for (s, t) in weights if s == current]
                if not neighbors:
                    break
                probs = np.array([weights[(current, t)] for t in neighbors])
                probs = probs / probs.sum()
                current = np.random.choice(neighbors, p=probs)
                walk.append(current)
            walks.append([str(s) for s in walk])
    return walks

def learn_embeddings(walks, feature_dim, window, min_count, sg):
    model = Word2Vec(
        sentences=walks,
        vector_size=feature_dim,
        window=window,
        min_count=min_count,
        sg=sg
    )
    return {int(k): model.wv[k] for k in model.wv.index_to_key}

def generate_scenario_features(
    data, n_trials=30, binary_fraction=0.3, seed=42,
    n_walks=20, walk_length=10,
    feature_dim=8, window=5, min_count=1, sg=1
):
    x_trials = generate_first_stage_trials(n_trials, data, binary_fraction, seed)
    duals = get_dual_vectors(x_trials, data)
    weights = compute_scenario_weights(duals, x_trials, list(data.M))
    walks = generate_random_walks(weights, list(data.M), n_walks, walk_length, seed)
    feature_vectors = learn_embeddings(walks, feature_dim, window, min_count, sg)
    return feature_vectors
