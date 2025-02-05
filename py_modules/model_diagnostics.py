import matplotlib.pyplot as plt

def plot_model_diagnostics(y_test, y_pred, model_name, 
                            title_residuals="Residuals Plot", 
                            title_actual_vs_pred="Actual vs Predicted"):
 
    residuals = y_test - y_pred
    plt.figure(figsize=(14, 6))  
    
    # Subplot 1: Residuals Plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, residuals, alpha=0.5, edgecolor='k', label="Residuals")
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f"{title_residuals} {model_name}", fontsize=14)
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Residuals", fontsize=12)
    plt.legend(fontsize=10)
    
    # Subplot 2: Actual vs Predicted Plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k', label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--', linewidth=1.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f"{title_actual_vs_pred} {model_name}", fontsize=14)
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()



