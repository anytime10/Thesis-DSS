def summarize_nested_cv_results(results, model_name="Model", save_metrics=False, metrics_file=None):
    """
    Summarizes and displays results from nested cross-validation.

    Parameters:
    - results (dict): The dictionary returned by nested_cross_validation.
    - model_name (str): Name of the model for display purposes.
    - save_metrics (bool): Whether to save the metrics DataFrame as a CSV file.
    - metrics_file (str): Filepath to save the metrics DataFrame if save_metrics is True.

    Returns:
    - None (displays key results and optionally saves the metrics DataFrame)
    """
    # Extract the metrics DataFrame
    outer_fold_metrics = results.get('outer_fold_metrics', None)
    if outer_fold_metrics is not None:
        print(f"\nOuter Fold Metrics for {model_name}:\n")
        print(outer_fold_metrics)
    else:
        print(f"\nNo metrics DataFrame available for {model_name}.\n")

    # Save the metrics DataFrame if requested
    if save_metrics and metrics_file and outer_fold_metrics is not None:
        outer_fold_metrics.to_csv(metrics_file, index=False)
        print(f"\nMetrics DataFrame saved to: {metrics_file}\n")

    # Display best model, metrics, and parameters
    best_model = results.get('best_model', None)
    best_metrics = results.get('best_metrics', {})
    best_params = results.get('best_params_list', [])[results.get('best_fold', 0)]

    print(f"Best Model for {model_name}:\n", best_model)
    print(f"Best Metrics ({model_name}):\n", best_metrics)
    print(f"Best Parameters ({model_name}):\n", best_params)

    # Return nothing but display everything
    return None
