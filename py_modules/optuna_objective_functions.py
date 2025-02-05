import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def objective_lasso(trial, X_train_inner, y_train_inner, groups_train_inner):
    alpha = trial.suggest_float('alpha', 1e-4, 1e1)
    model = Lasso(alpha=alpha, random_state=42)
    
    gkf_inner = GroupKFold(n_splits=5)
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    # Perform cross-validation
    for train_idx, val_idx in gkf_inner.split(X_train_inner, y_train_inner, groups_train_inner):
        X_tr, X_val = X_train_inner.iloc[train_idx], X_train_inner.iloc[val_idx]
        y_tr, y_val = y_train_inner.iloc[train_idx], y_train_inner.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
    
    return np.mean(rmse_scores), np.mean(mae_scores), np.mean(r2_scores)


def objective_rf(trial, X_train_inner, y_train_inner, groups_train_inner):
    # Define the hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 5, 200)
    max_depth = trial.suggest_int('max_depth', 3, 70)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    # initializing the model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features= max_features,
        random_state=42,
        n_jobs=-1
    )
    
    gkf_inner = GroupKFold(n_splits=5)
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    # Perform cross-validation
    for train_idx, val_idx in gkf_inner.split(X_train_inner, y_train_inner, groups_train_inner):
        X_tr, X_val = X_train_inner.iloc[train_idx], X_train_inner.iloc[val_idx]
        y_tr, y_val = y_train_inner.iloc[train_idx], y_train_inner.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
    
    return np.mean(rmse_scores), np.mean(mae_scores), np.mean(r2_scores)