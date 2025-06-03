import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from config import *
from gurobipy import Model, GRB, quicksum

def train_prediction_function(
    dual_vectors,
    feature_vectors,
    method=PREDICTION_METHOD,
    n_neighbors=N_NEIGHBORS,
    seed=SEED,
    customized_prediction_model=False,
    max_loss_L=None
):
    """
    Train a prediction function to estimate dual vectors for each scenario based on its feature vector.

    This function supports both:
    - Standard machine learning models (linear regression and KNN), and
    - A custom constrained linear regression model trained using Gurobi, as described in the
      TR(\bar{x}) formulation with a zero objective and bounded absolute prediction error.

    Parameters:
        dual_vectors (dict): Mapping from scenario ID to dual vector (numpy array of length n_j)
        feature_vectors (dict): Mapping from scenario ID to learned feature vector (numpy array)
        method (str): Prediction method; must be 'regression' or 'knn'
        n_neighbors (int): Number of neighbors for KNN (if used)
        seed (int): Random seed for reproducibility
        customized_prediction_model (bool): If True and method is 'regression', trains constrained model using Gurobi
        max_loss_L (float or None): If provided, sets an upper bound on L during training. If None, L is free.

    Returns:
        predictor (callable): A function that accepts a 2D array X_test and returns predicted dual vectors
    """
    np.random.seed(seed)
    selected_scenarios = list(dual_vectors.keys())

    # Prepare training input and target matrices
    X_train = np.array([feature_vectors[s] for s in selected_scenarios])  # Features for selected scenarios
    Y_train = np.array([np.array(dual_vectors[s], dtype=float) for s in selected_scenarios])  # Dual vectors

    if method == "regression":
        if customized_prediction_model:
            def fit_constrained_regression(X, Y, L_bound=None):
                """
                Fit a linear model using constrained optimization:
                    min 0
                    s.t. max absolute error across all output dimensions <= L

                Returns a predictor function based on learned theta matrix.
                """
                num_samples, num_features = X.shape
                output_dim = Y.shape[1]

                model = Model("Custom_Constrained_Regression")
                model.Params.OutputFlag = 0  # Turn off Gurobi output

                # Create theta[i, j]: coefficient for feature j in output dimension i
                theta = model.addVars(output_dim, num_features, lb=-GRB.INFINITY, name="theta")

                # Loss variable L bounding the maximum absolute prediction error
                L = model.addVar(lb=0, name="L")

                # If user specifies a max loss, enforce it
                if L_bound is not None:
                    model.addConstr(L <= L_bound, name="MaxLossBound")

                # Add constraints for each training example and output dimension
                for m in range(num_samples):
                    x = X[m, :]
                    y_true = Y[m, :]

                    for i in range(output_dim):
                        pred = quicksum(theta[i, j] * x[j] for j in range(num_features))  # Linear prediction
                        error = pred - y_true[i]  # Prediction error
                        model.addConstr(error <= L)  # Constraint: upper bound on error
                        model.addConstr(-error <= L)  # Constraint: lower bound on error (symmetry)

                model.setObjective(0, GRB.MINIMIZE)  # Dummy objective: just feasibility
                model.optimize()  # Solve the model

                # Extract trained theta matrix from solution
                theta_values = np.array([[theta[i, j].X for j in range(num_features)] for i in range(output_dim)])

                def predictor(X_test):
                    """Return predicted dual vectors using learned theta matrix."""
                    return np.dot(X_test, theta_values.T)

                return predictor

            # Train the custom model using Gurobi
            predictor = fit_constrained_regression(X_train, Y_train, L_bound=max_loss_L)

        else:
            # Standard linear regression using scikit-learn
            model = LinearRegression().fit(X_train, Y_train)
            predictor = model.predict

    elif method == "knn":
        # K-Nearest Neighbors regression using scikit-learn
        model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, Y_train)
        predictor = model.predict

    else:
        raise ValueError("Invalid method. Choose 'regression' or 'knn'.")

    return predictor