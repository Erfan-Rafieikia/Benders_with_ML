import numpy as np
from config import *

def generate_random_walks(weight_values, n_walk=N_WALK, l_walk=L_WALK, seed=SEED):
    """
    Generate random walks for each follower scenario s using probability π_st.
    """
    np.random.seed(seed)  # Ensuring reproducibility

    C = []  # Walk container

    scenarios = list(set([s for s, _ in weight_values.keys()]))

    for i in range(n_walk):  # Number of walks per scenario
        for s in scenarios:  # Iterate over all scenarios
            walk = [s]  # Start at scenario s
            v_curr = s

            for j in range(l_walk):  # Walk length
                candidates = [t for (s_i, t) in weight_values.keys() if s_i == v_curr]

                if not candidates:  # If no transitions are possible, break
                    break

                # Get transition probabilities
                probabilities = np.array([weight_values[(v_curr, t)] for t in candidates])
                probabilities /= probabilities.sum()  # Normalize

                # Sample next scenario based on probabilities
                v_next = np.random.choice(candidates, p=probabilities)

                walk.append(v_next)
                v_curr = v_next

            C.append(walk)  # Store the walk

    return C