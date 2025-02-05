import matplotlib.pyplot as plt
import numpy as np


def plot_merf_convergence(merf, b_hat_hist):
    # Calculate mean variance across clusters and variances by cluster
    sigma_b2_history = [np.mean(np.diag(D)) for D in merf.D_hat_history]  # Mean variance across clusters
    cluster_variances = np.array([np.diag(D) for D in merf.D_hat_history]).T  # Variances by cluster
    
    # Pivot b_hat_history DataFrame for plotting
    b_hat_pivot = b_hat_hist.pivot(index='iteration', columns='cluster', values='beta')

    # Create the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("MERF Model Convergence", fontsize=16)

    # Subplot 1: GLL History
    axes[0, 0].plot(range(len(merf.gll_history)), merf.gll_history, linestyle='-')
    axes[0, 0].set_title("GLL History")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("GLL")
    axes[0, 0].grid(True)

    # Subplot 2: Random Effects Variance (sigma_b2)
    for i, cluster_variance in enumerate(cluster_variances):
        axes[0, 1].plot(range(len(cluster_variance)), cluster_variance)
    axes[0, 1].set_title("Random Effects Variance (sigma_b2)")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("sigma_b2")
    axes[0, 1].grid(True)

    # Subplot 3: Residual Variance (sigma2_hat)
    axes[1, 0].plot(range(len(merf.sigma2_hat_history)), merf.sigma2_hat_history, marker='o', linestyle='-')
    axes[1, 0].set_title("Residual Variance (sigma2_hat)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("sigma2_hat")
    axes[1, 0].grid(True)

    # Subplot 4: b_hat Convergence
    for cluster in b_hat_pivot.columns:
        axes[1, 1].plot(b_hat_pivot.index, b_hat_pivot[cluster], label=cluster)
    axes[1, 1].set_title("b_hat Convergence")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("b_hat")
    axes[1, 1].legend(title="Cluster")
    axes[1, 1].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()


def fast_permutation_importance_merf(merf, X_test, y_test, clusters_test):
    # Base residuals and variance
    base_predictions = merf.predict(X_test, np.ones((X_test.shape[0], 1)), clusters_test)
    base_residuals = y_test - base_predictions
    base_variance = np.var(base_residuals, ddof=1)

    feature_importances = []
    for feature in X_test.columns:
        # Shuffle feature
        X_shuffled = X_test.copy()
        X_shuffled[feature] = np.random.permutation(X_shuffled[feature])

        # Predict with shuffled feature
        shuffled_predictions = merf.predict(X_shuffled, np.ones((X_shuffled.shape[0], 1)), clusters_test)
        shuffled_residuals = y_test - shuffled_predictions
        shuffled_variance = np.var(shuffled_residuals, ddof=1)

        # Importance: Increase in residual variance
        importance = shuffled_variance - base_variance
        feature_importances.append((feature, importance))

    return sorted(feature_importances, key=lambda x: x[1], reverse=True)