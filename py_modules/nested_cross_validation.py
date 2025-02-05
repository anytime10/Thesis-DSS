import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import pandas as pd
from merf import MERF
from sklearn.ensemble import RandomForestRegressor

def nested_cross_validation(
    objective_function, X, y, groups, model_class, n_trials=100, outer_splits=5, metric="rmse"
):
    outer_rmse_scores = []
    outer_mae_scores = []
    outer_r2_scores = []
    study_list = []
    best_params_list = []
    best_trials = []
    trained_models = []
    important_features_list = []  # To store important features for each fold

    gkf_outer = GroupKFold(n_splits=outer_splits)

    for outer_fold, (train_idx, val_idx) in enumerate(gkf_outer.split(X, y, groups)):
        # Outer training and validation splits
        X_train_outer = X.iloc[train_idx]
        X_val_outer = X.iloc[val_idx]
        y_train_outer = y.iloc[train_idx]
        y_val_outer = y.iloc[val_idx]
        groups_train_outer = groups.iloc[train_idx]

        # Inner optimization with Optuna
        study = optuna.create_study(directions=["minimize", "minimize", "maximize"])  # RMSE, MAE, R²
        study.optimize(
            lambda trial: objective_function(trial, X_train_outer, y_train_outer, groups_train_outer),
            n_trials=n_trials
        )
        study_list.append(study)

        # Get the best trial (based on a desired metric, e.g., lowest RMSE)
        best_trial = sorted(study.best_trials, key=lambda t: t.values[0])[0]  # Sort by RMSE (1st metric)
        best_trials.append(best_trial)
        best_params = best_trial.params
        best_params_list.append(best_params)

        # Train the model with the best hyperparameters
        if 'n_jobs' in model_class.__init__.__code__.co_varnames:
            # Include n_jobs if the model supports it
            model = model_class(**best_params, random_state=42, n_jobs=-1)
        else:
            # Exclude n_jobs for models that do not support it
            model = model_class(**best_params, random_state=42)

        model.fit(X_train_outer, y_train_outer)
        trained_models.append(model)

        # Evaluate on the outer validation set
        y_pred_outer = model.predict(X_val_outer)
        rmse_outer = np.sqrt(mean_squared_error(y_val_outer, y_pred_outer))
        mae_outer = mean_absolute_error(y_val_outer, y_pred_outer)
        r2_outer = r2_score(y_val_outer, y_pred_outer)

        # Append metrics for this fold
        outer_rmse_scores.append(rmse_outer)
        outer_mae_scores.append(mae_outer)
        outer_r2_scores.append(r2_outer)

        # Identify important features (non-zero coefficients for Lasso)
        if hasattr(model, 'coef_'):  # Check if the model has coefficients
            coefficients = model.coef_
            feature_names = X_train_outer.columns  # Feature names
            important_features = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            }).query("Coefficient != 0")  # Filter non-zero coefficients
            important_features_list.append(important_features)
        elif hasattr(model, 'feature_importances_'):  # For models like Random Forest
            feature_importances = model.feature_importances_
            feature_names = X_train_outer.columns  # Feature names
            important_features = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).query("Importance > 0")  # Filter features with non-zero importance
            important_features_list.append(important_features)
        else:
            important_features_list.append(None) 

    # Determine the best fold based on the selected metric
    if metric == "rmse":
        best_fold = np.argmin(outer_rmse_scores)
    elif metric == "mae":
        best_fold = np.argmin(outer_mae_scores)
    elif metric == "r2":
        best_fold = np.argmax(outer_r2_scores)
    else:
        raise ValueError("Invalid metric. Choose from 'rmse', 'mae', or 'r2'.")

    best_model = trained_models[best_fold]
    best_metrics = {
        "rmse": outer_rmse_scores[best_fold],
        "mae": outer_mae_scores[best_fold],
        "r2": outer_r2_scores[best_fold],
    }

    # Create a DataFrame summarizing all outer fold metrics
    outer_fold_metrics = pd.DataFrame({
        'Fold': range(1, outer_splits + 1),
        'RMSE': outer_rmse_scores,
        'MAE': outer_mae_scores,
        'R2': outer_r2_scores,
        'Best_Params': best_params_list
    })

    # Add an indicator for the best fold
    outer_fold_metrics['Best_Fold'] = outer_fold_metrics['Fold'] == (best_fold + 1)

    # Compile results into a dictionary
    results = {
        'rmse_scores': outer_rmse_scores,
        'mae_scores': outer_mae_scores,
        'r2_scores': outer_r2_scores,
        'study_list': study_list,
        'best_params_list': best_params_list,
        'best_trials': best_trials,
        'models': trained_models,
        'best_model': best_model,
        'best_fold': best_fold,
        'best_metrics': best_metrics,
        'outer_fold_metrics': outer_fold_metrics,  # Include the DataFrame in results
        'important_features_list': important_features_list  # Add important features
    }

    return results

def nested_cross_validation_merf(
    X, y, Z, clusters, groups,
    n_trials=50,
    outer_splits=5,
    metric="rmse"
):
    """
    Nested cross-validation for MERF with pre-tuned RF hyperparameters and early stopping.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import optuna
    import numpy as np
    from merf import MERF
    from sklearn.ensemble import RandomForestRegressor

    # Pre-tuned Random Forest hyperparameters for the fixed effects model
    rf_fixed_effects = RandomForestRegressor(
        n_estimators=92,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=9,
        random_state=42
    )

    # Outer loop performance metrics
    outer_rmse_scores = []
    outer_mae_scores = []
    outer_r2_scores = []
    best_trials = []
    trained_merf_models = []

    # Outer GroupKFold for cross-validation
    gkf_outer = GroupKFold(n_splits=outer_splits)

    for train_idx, test_idx in gkf_outer.split(X, y, groups):
        # Outer train-test split
        X_train_outer = X.iloc[train_idx].reset_index(drop=True)
        X_test_outer = X.iloc[test_idx].reset_index(drop=True)
        y_train_outer = y.iloc[train_idx].reset_index(drop=True)
        y_test_outer = y.iloc[test_idx].reset_index(drop=True)
        Z_train_outer = Z[train_idx]  # NumPy arrays don’t require reset_index
        Z_test_outer = Z[test_idx]
        clusters_train_outer = clusters.iloc[train_idx].reset_index(drop=True)
        clusters_test_outer = clusters.iloc[test_idx].reset_index(drop=True)
        groups_train_outer = groups.iloc[train_idx].reset_index(drop=True)

        # Inner loop hyperparameter tuning with Optuna
        def objective(trial):
            # Suggest MERF-specific hyperparameters
            max_iterations = trial.suggest_int("max_iterations", 10, 50)
            gll_threshold = trial.suggest_float("gll_threshold", 1e-4, 1e-2, log=True)

            # Initialize MERF with pre-tuned RF and suggested MERF hyperparameters
            merf = MERF(
                fixed_effects_model=rf_fixed_effects,
                max_iterations=max_iterations,
                gll_early_stop_threshold=gll_threshold
            )

            # Set up GroupKFold for inner cross-validation
            gkf_inner = GroupKFold(n_splits=5)

            rmse_scores = []
            mae_scores = []
            r2_scores = []

            # Perform inner cross-validation
            for inner_train_idx, inner_val_idx in gkf_inner.split(X_train_outer, y_train_outer, groups_train_outer):
                # Split inner train and validation sets with index reset
                X_train_inner = X_train_outer.iloc[inner_train_idx].reset_index(drop=True)
                X_val_inner = X_train_outer.iloc[inner_val_idx].reset_index(drop=True)
                y_train_inner = y_train_outer.iloc[inner_train_idx].reset_index(drop=True)
                y_val_inner = y_train_outer.iloc[inner_val_idx].reset_index(drop=True)
                Z_train_inner = Z_train_outer[inner_train_idx]
                Z_val_inner = Z_train_outer[inner_val_idx]
                clusters_train_inner = clusters_train_outer.iloc[inner_train_idx].reset_index(drop=True)
                clusters_val_inner = clusters_train_outer.iloc[inner_val_idx].reset_index(drop=True)

                # Fit MERF
                merf.fit(X_train_inner, Z_train_inner, clusters_train_inner, y_train_inner)

                # Predict on validation set
                y_pred_inner = merf.predict(X_val_inner, Z_val_inner, clusters_val_inner)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_val_inner, y_pred_inner))
                mae = mean_absolute_error(y_val_inner, y_pred_inner)
                r2 = r2_score(y_val_inner, y_pred_inner)

                # Append metrics for this fold
                rmse_scores.append(rmse)
                mae_scores.append(mae)
                r2_scores.append(r2)

            # Return average RMSE, MAE, and R² across folds
            return np.mean(rmse_scores), np.mean(mae_scores), np.mean(r2_scores)

        # Optimize with Optuna
        study = optuna.create_study(directions=["minimize", "minimize", "maximize"])  # RMSE, MAE, R²
        study.optimize(objective, n_trials=n_trials)

        # Get best hyperparameters and train final MERF on outer train set
        best_trial = sorted(study.best_trials, key=lambda t: t.values[0])[0]
        best_trials.append(best_trial)
        best_params = best_trial.params

        # Train MERF on outer train set with best parameters
        merf = MERF(
            fixed_effects_model=rf_fixed_effects,
            max_iterations=best_params["max_iterations"],
            gll_early_stop_threshold=best_params["gll_threshold"]
        )
        merf.fit(X_train_outer, Z_train_outer, clusters_train_outer, y_train_outer)
        trained_merf_models.append(merf)

        # Evaluate MERF on the outer test set
        y_pred_outer = merf.predict(X_test_outer, Z_test_outer, clusters_test_outer)
        rmse = np.sqrt(mean_squared_error(y_test_outer, y_pred_outer))
        mae = mean_absolute_error(y_test_outer, y_pred_outer)
        r2 = r2_score(y_test_outer, y_pred_outer)

        # Append metrics for this fold
        outer_rmse_scores.append(rmse)
        outer_mae_scores.append(mae)
        outer_r2_scores.append(r2)

    # Determine the best fold based on the selected metric
    if metric == "rmse":
        best_fold = np.argmin(outer_rmse_scores)
    elif metric == "mae":
        best_fold = np.argmin(outer_mae_scores)
    elif metric == "r2":
        best_fold = np.argmax(outer_r2_scores)
    else:
        raise ValueError("Invalid metric. Choose from 'rmse', 'mae', or 'r2'.")

    best_merf_model = trained_merf_models[best_fold]
    best_metrics = {
        "rmse": outer_rmse_scores[best_fold],
        "mae": outer_mae_scores[best_fold],
        "r2": outer_r2_scores[best_fold],
    }

    # Compile results
    results = {
        'rmse_scores': outer_rmse_scores,
        'mae_scores': outer_mae_scores,
        'r2_scores': outer_r2_scores,
        'best_trials': best_trials,
        'trained_merf_models': trained_merf_models,
        'best_merf_model': best_merf_model,
        'best_fold': best_fold,
        'best_metrics': best_metrics
    }

    return results



