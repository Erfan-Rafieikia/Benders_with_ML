"""
Callback module for Benders decomposition with dual-based cuts
in a general two-stage stochastic MILP with optional ML guidance.

Overview:
---------
This callback implements the logic for generating optimality cuts dynamically
during the branch-and-bound process using Gurobi's lazy constraint callback (MIPSOL).
Cuts are derived from the dual of the second-stage linear program for each scenario.

Key Features:
-------------
1. **Cut Generation for Selected Scenarios**:
   - For a predefined subset of scenarios, solves the dual subproblem exactly
     and generates standard Benders optimality cuts of the form:
       λᵗ(Bx - b) ≤ η

2. **ML-Based Cut Approximation for Unselected Scenarios** (Optional):
   - If `use_prediction=True`, the dual vectors from selected scenarios are used
     to train a predictive model (e.g., regression, KNN, etc.).
   - The trained model is used to estimate dual vectors for unselected scenarios.
   - If prediction fails (e.g., due to infeasible regression model), the dual is
     computed exactly instead.

3. **Fractional x Handling**:
   - Controlled by `allow_fractional_x`. If False, cuts are only generated when
     the first-stage variables are integral. Otherwise, fractional solutions
     are also cut.

4. **Cut Categorization and Tracking**:
   - Tracks how many cuts are generated in each category:
     - Selected scenarios (MIP/relaxed)
     - Unselected scenarios using prediction (MIP/relaxed)
     - Unselected scenarios using solving (MIP/relaxed)

5. **Training Model Reliability Tracking**:
   - Tracks the success rate of predictive model training across iterations.
   - Reports percentage of iterations where ML model training was successful.

Typical Usage:
--------------
This callback is meant to be passed to a Gurobi model via `model.optimize(callback)`,
typically as part of a Benders decomposition where the master problem handles
first-stage decisions and second-stage costs are approximated by η variables.

Dependencies:
-------------
- Requires `solve_dual_subproblem_general()` to solve the dual LP.
- Requires `train_dual_model()` and `predict_duals()` for ML-guided prediction.

Intended for use in:
---------------------
- Two-stage stochastic integer programming models with dual decomposition
- ML-augmented Benders decomposition
- Academic and research applications in optimization

"""



from gurobipy import GRB
from subproblem_dual import solve_dual_subproblem_general
from train_predict_dual import train_dual_model, predict_duals


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
        """
        Custom callback for general two-stage stochastic MILP with dual cuts.

        Parameters:
        ----------
        dat : DataGeneral
            Instance of the data object with model components.
        x_vars : gurobipy.VarDict
            First-stage decision variables (indexed by J).
        eta_vars : gurobipy.VarDict
            Second-stage recourse approximation variables (indexed by scenario m).
        selected_scenarios : list[int]
            Scenario indices used for training and direct solving.
        feature_vectors : dict[int, np.ndarray]
            Feature vectors used for training/prediction.
        prediction_method : str
            Method for ML prediction ('knn', 'tree', or 'regression').
        n_neighbors : int
            Number of neighbors (if applicable).
        use_prediction : bool
            Whether to use ML-based dual prediction for unselected scenarios.
        allow_fractional_x : bool
            Whether to generate cuts even when x is fractional.
        """
        self.dat = dat
        self.x_vars = x_vars
        self.eta_vars = eta_vars
        self.selected = selected_scenarios
        self.features = feature_vectors
        self.predict_method = prediction_method
        self.k = n_neighbors
        self.use_prediction = use_prediction
        self.allow_fractional_x = allow_fractional_x

        # Tracking cut counts by category
        self.num_cuts_mip_selected = {}
        self.num_cuts_rel_selected = {}
        self.num_cuts_mip_ml = {}
        self.num_cuts_rel_ml = {}
        self.num_cuts_mip_unselected = {}
        self.num_cuts_rel_unselected = {}

        # Counters to track how often training fails
        self.total_iterations = 0
        self.training_successes = 0

    def __call__(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return

        # Check if x is integer-valued unless fractional is allowed
        x_val = {j: model.cbGetSolution(self.x_vars[j]) for j in self.dat.J}
        if not self.allow_fractional_x and any(abs(x_val[j] - round(x_val[j])) > 1e-5 for j in self.dat.J):
            return

        self.total_iterations += 1

        # ──────────────────────────────────────────────
        # Solve dual subproblem for selected scenarios
        # ──────────────────────────────────────────────
        duals_selected = {}
        for m in self.selected:
            obj, dual_vec = solve_dual_subproblem_general(x_val, m, self.dat)
            duals_selected[m] = dual_vec

            # Construct Benders cut: λᵗ(Bx - b) ≤ η
            expr = sum(
                dual_vec[i] * sum(self.dat.A_matrices[m][i, j] * self.x_vars[j] for j in self.dat.J)
                for i in self.dat.I
            ) - sum(dual_vec[i] * self.dat.rhs_vectors[m][i] for i in self.dat.I)
            model.cbLazy(expr <= self.eta_vars[m])

            # Log type of cut (MIP vs relaxed)
            cut_type = self.num_cuts_mip_selected if self._is_integral(model) else self.num_cuts_rel_selected
            cut_type[m] = cut_type.get(m, 0) + 1

        # ──────────────────────────────────────────────
        # ML prediction for unselected scenarios
        # ──────────────────────────────────────────────
        if self.use_prediction:
            try:
                trained_model = train_dual_model(
                    scenarios=self.selected,
                    features=self.features,
                    duals=duals_selected,
                    method=self.predict_method,
                    k=self.k
                )
                if trained_model is None:
                    raise ValueError("Training model is None")

                self.training_successes += 1

                pred_duals = predict_duals(
                    trained_model, self.features, self.dat.M, exclude=self.selected
                )

                for m, dual_vec in pred_duals.items():
                    expr = sum(
                        dual_vec[i] * sum(self.dat.A_matrices[m][i, j] * self.x_vars[j] for j in self.dat.J)
                        for i in self.dat.I
                    ) - sum(dual_vec[i] * self.dat.rhs_vectors[m][i] for i in self.dat.I)
                    model.cbLazy(expr <= self.eta_vars[m])

                    cut_type = self.num_cuts_mip_ml if self._is_integral(model) else self.num_cuts_rel_ml
                    cut_type[m] = cut_type.get(m, 0) + 1

            except Exception as e:
                print(f"For x = {x_val}, training model doesn't exist due to: {e}. Falling back to solving.")

                for m in self.dat.M:
                    if m in self.selected:
                        continue
                    obj, dual_vec = solve_dual_subproblem_general(x_val, m, self.dat)
                    expr = sum(
                        dual_vec[i] * sum(self.dat.A_matrices[m][i, j] * self.x_vars[j] for j in self.dat.J)
                        for i in self.dat.I
                    ) - sum(dual_vec[i] * self.dat.rhs_vectors[m][i] for i in self.dat.I)
                    model.cbLazy(expr <= self.eta_vars[m])

                    cut_type = self.num_cuts_mip_unselected if self._is_integral(model) else self.num_cuts_rel_unselected
                    cut_type[m] = cut_type.get(m, 0) + 1

        # ──────────────────────────────────────────────
        # No ML: solve all unselected scenarios
        # ──────────────────────────────────────────────
        else:
            for m in self.dat.M:
                if m in self.selected:
                    continue
                obj, dual_vec = solve_dual_subproblem_general(x_val, m, self.dat)
                expr = sum(
                    dual_vec[i] * sum(self.dat.A_matrices[m][i, j] * self.x_vars[j] for j in self.dat.J)
                    for i in self.dat.I
                ) - sum(dual_vec[i] * self.dat.rhs_vectors[m][i] for i in self.dat.I)
                model.cbLazy(expr <= self.eta_vars[m])

                cut_type = self.num_cuts_mip_unselected if self._is_integral(model) else self.num_cuts_rel_unselected
                cut_type[m] = cut_type.get(m, 0) + 1

        # Print training model success rate after final iteration
        if self.total_iterations > 0 and model.cbGet(GRB.Callback.MIPSOL_OBJBND) == model.cbGet(GRB.Callback.MIPSOL_OBJ):
            rate = self.training_successes / self.total_iterations * 100
            print(f"Training model success rate: {rate:.2f}% over {self.total_iterations} x iterations.")

    def _is_integral(self, model):
        """Check whether the current x solution is integral-valued."""
        return all(abs(model.cbGetSolution(self.x_vars[j]) - round(model.cbGetSolution(self.x_vars[j]))) <= 1e-5 for j in self.dat.J)
