import pandas as pd

def get_best_model_features(important_features_list, best_fold_index):
   
    # Retrieve the important features for the best fold
    important_features_best_model = important_features_list[best_fold_index]

    if important_features_best_model is not None:
        # Check if the model provides coefficients or importances
        if 'Coefficient' in important_features_best_model.columns:
            value_column = 'Coefficient'  # For Lasso or linear models
        elif 'Importance' in important_features_best_model.columns:
            value_column = 'Importance'  # For Random Forest, MERF, etc.
        else:
            raise ValueError("The DataFrame must contain either 'Coefficient' or 'Importance' column.")

        # Create a DataFrame for the best-performing model
        best_model_features_df = pd.DataFrame({
            'Feature': important_features_best_model['Feature'],
            value_column: important_features_best_model[value_column]
        })

        # Sort by absolute value (optional, for interpretability)
        best_model_features_df = best_model_features_df.sort_values(by=value_column, ascending=False, key=abs)

        return best_model_features_df
    else:
        # Return an empty DataFrame if no important features were identified
        return pd.DataFrame(columns=['Feature', 'Value'])
