import time
import numpy as np
from gurobipy import GRB, quicksum
from subproblem_dual import solve_dual_subproblem_general
from training_function import train_dual_model, predict_duals


class Callback:
    def __init__(
        self,
        dat,
        x_vars,
        eta_vars,
        selected_scenarios,
        feature_vectors,
        prediction_method,
        n_neighbors=5,
        use_prediction=True,
        allow_fractional_x=True
    ):
        self.dat = dat
        self.x_vars = x_vars
        self.eta_vars = eta_vars
        self.selected = selected_scenarios
        self.features = feature_vectors
        self.predict_method = prediction_method
        self.k = n_neighbors
        self.use_prediction = use_prediction
        self.allow_fractional_x = allow_fractional_x

        # For timing
        self.time_training = 0.0
        self.time_solving_unselected = 0.0

        # For counting cuts
        self.num_cuts_mip_selected = {}
        self.num_cuts_rel_selected = {}
        self.num_cuts_mip_ml = {}
        self.num_cuts_rel_ml = {}
        self.num_cuts_mip_unselected = {}
        self.num_cuts_rel_unselected = {}

        # Internal model access
        self.model = None

    def __call__(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return

        self.model = model

        x_val = model.cbGetSolution(self.x_vars)
        is_integral = all(abs(x_val[j] - round(x_val[j])) <= 1e-5 for j in self.dat.J)

        if not self.allow_fractional_x and not is_integral:
            return

        # Solve selected subproblems to get duals
        duals_selected = {}
        for m in self.selected:
            _, dual_vec = solve_dual_subproblem_general(x_val, m, self.dat)
            duals_selected[m] = np.array([dual_vec[i] for i in self.dat.I])  # Convert dict to array

            expr = sum(
                dual_vec[i] * sum(self.dat.A_matrices[m][i, j] * self.x_vars[j] for j in self.dat.J)
                for i in self.dat.I
            ) - sum(dual_vec[i] * self.dat.rhs_vectors[m][i] for i in self.dat.I)

            model.cbLazy(expr <= self.eta_vars[m])

            cut_type = self.num_cuts_mip_selected if is_integral else self.num_cuts_rel_selected
            cut_type[m] = cut_type.get(m, 0) + 1

        # Train model
        start = time.time()
        trained_model = train_dual_model(
            scenarios=self.selected,
            features=self.features,
            duals=duals_selected,
            method=self.predict_method,
            k=self.k,
            use_customized=True
        )
        self.time_training += time.time() - start

        # Predict for others
        if self.use_prediction and trained_model is not None:
            pred_duals = predict_duals(trained_model, self.features, list(self.dat.M), exclude=self.selected)
            for m, dual_vec in pred_duals.items():
                expr = sum(
                    dual_vec[i] * sum(self.dat.A_matrices[m][i, j] * self.x_vars[j] for j in self.dat.J)
                    for i in self.dat.I
                ) - sum(dual_vec[i] * self.dat.rhs_vectors[m][i] for i in self.dat.I)

                model.cbLazy(expr <= self.eta_vars[m])

                cut_type = self.num_cuts_mip_ml if is_integral else self.num_cuts_rel_ml
                cut_type[m] = cut_type.get(m, 0) + 1

        # Fallback: solve the rest subproblems if necessary
        self._solve_fallback(model, x_val, is_integral)

    def _solve_fallback(self, model, x_val, is_integral):
        """Generate cuts for unselected scenarios (fallback if prediction is disabled or fails)."""
        start = time.time()
        for m in self.dat.M:
            if m in self.selected:
                continue

            _, dual_vec = solve_dual_subproblem_general(x_val, m, self.dat)

            expr = sum(
                dual_vec[i] * sum(self.dat.A_matrices[m][i, j] * self.x_vars[j] for j in self.dat.J)
                for i in self.dat.I
            ) - sum(dual_vec[i] * self.dat.rhs_vectors[m][i] for i in self.dat.I)

            model.cbLazy(expr <= self.eta_vars[m])

            cut_type = self.num_cuts_mip_unselected if is_integral else self.num_cuts_rel_unselected
            cut_type[m] = cut_type.get(m, 0) + 1
        self.time_solving_unselected += time.time() - start
