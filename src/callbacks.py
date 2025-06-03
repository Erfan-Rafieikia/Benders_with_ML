import time
import numpy as np
from gurobipy import GRB
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
        n_neighbors,
        use_prediction,
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

        self.num_cuts_mip_selected = {}
        self.num_cuts_rel_selected = {}
        self.num_cuts_mip_ml = {}
        self.num_cuts_rel_ml = {}
        self.num_cuts_mip_unselected = {}
        self.num_cuts_rel_unselected = {}

        self.total_iterations = 0
        self.training_successes = 0

        self.time_training = 0
        self.time_prediction = 0
        self.time_solving_selected = 0
        self.time_solving_unselected = 0

    def __call__(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return

        x_val = {j: model.cbGetSolution(self.x_vars[j]) for j in self.dat.J}
        is_integral = all(abs(x_val[j] - round(x_val[j])) <= 1e-5 for j in self.dat.J)
        if not self.allow_fractional_x and not is_integral:
            return

        self.total_iterations += 1

        # Selected scenario cuts (exact)
        start_selected = time.time()
        duals_selected = {}
        for m in self.selected:
            _, dual_vec = solve_dual_subproblem_general(x_val, m, self.dat)
            # Convert to NumPy array
            dual_vec_np = np.array([dual_vec[i] for i in self.dat.I])
            duals_selected[m] = dual_vec_np

            expr = sum(
                dual_vec[i] * sum(self.dat.A_matrices[m][i, j] * self.x_vars[j] for j in self.dat.J)
                for i in self.dat.I
            ) - sum(dual_vec[i] * self.dat.rhs_vectors[m][i] for i in self.dat.I)
            model.cbLazy(expr <= self.eta_vars[m])
            cut_dict = self.num_cuts_mip_selected if is_integral else self.num_cuts_rel_selected
            cut_dict[m] = cut_dict.get(m, 0) + 1
        self.time_solving_selected += time.time() - start_selected

        # ML predictions
        if self.use_prediction:
            start_train = time.time()
            try:
                trained_model = train_dual_model(
                    scenarios=self.selected,
                    features=self.features,
                    duals=duals_selected,
                    method=self.predict_method,
                    k=self.k
                )
                if trained_model is None:
                    raise ValueError("Training model returned None")
                self.training_successes += 1
            except Exception as e:
                print(f"[Callback Warning] Training failed: {e}")
                trained_model = None
            self.time_training += time.time() - start_train

            if trained_model:
                start_pred = time.time()
                pred_duals = predict_duals(
                    trained_model,
                    self.features,
                    list(self.dat.M),
                    exclude=self.selected
                )
                for m, dual_vec in pred_duals.items():
                    expr = sum(
                        dual_vec[i] * sum(self.dat.A_matrices[m][i, j] * self.x_vars[j] for j in self.dat.J)
                        for i in self.dat.I
                    ) - sum(dual_vec[i] * self.dat.rhs_vectors[m][i] for i in self.dat.I)
                    model.cbLazy(expr <= self.eta_vars[m])
                    cut_dict = self.num_cuts_mip_ml if is_integral else self.num_cuts_rel_ml
                    cut_dict[m] = cut_dict.get(m, 0) + 1
                self.time_prediction += time.time() - start_pred
            else:
                self._solve_fallback(model, x_val, is_integral)
        else:
            self._solve_fallback(model, x_val, is_integral)

        # Final summary if optimal
        if self.total_iterations > 0 and model.cbGet(GRB.Callback.MIPSOL_OBJBND) == model.cbGet(GRB.Callback.MIPSOL_OBJ):
            rate = self.training_successes / self.total_iterations * 100
            print(f"\n[Callback Summary]")
            print(f"Training success rate: {rate:.1f}%")
            print(f"Time - training: {self.time_training:.2f}s | prediction: {self.time_prediction:.2f}s")
            print(f"Time - solving selected: {self.time_solving_selected:.2f}s | unselected: {self.time_solving_unselected:.2f}s")

    def _solve_fallback(self, model, x_val, is_integral):
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
            cut_dict = self.num_cuts_mip_unselected if is_integral else self.num_cuts_rel_unselected
            cut_dict[m] = cut_dict.get(m, 0) + 1
        self.time_solving_unselected += time.time() - start
