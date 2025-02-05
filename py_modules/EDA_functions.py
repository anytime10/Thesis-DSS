import pandas as pd  
import numpy as np  
from scipy.stats import chi2_contingency, f_oneway  
from sklearn.metrics import mutual_info_score  
from scipy.stats import kruskal
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def categorical_associations(df, categorical_cols):
    results = []

    # Iterate through each pair of categorical variables
    for i, var1 in enumerate(categorical_cols):
        for var2 in categorical_cols[i + 1:]:  # Avoid duplicate comparisons
            # Create a contingency table
            contingency_table = pd.crosstab(df[var1], df[var2])

            # Perform the Chi-Square test
            try:
                chi2, p, _, _ = chi2_contingency(contingency_table)
                results.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Chi-Square Statistic': chi2,
                    'P-Value': p
                })
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Chi-Square Statistic': None,
                    'P-Value': None,
                    'Error': str(e)
                })

    # Return the results as a DataFrame for easy viewing
    return pd.DataFrame(results)



def cat_cont_kruskal(df, continuos_cols, categorical_cols):
    results = []

    for cat in categorical_cols:
        for cont in continuos_cols:
            # Drop rows with missing values for these columns
            valid_rows = df[[cat, cont]].dropna()

            if valid_rows.shape[0] > 1:  # Ensure there's enough data
                unique_groups = valid_rows[cat].nunique()

                if unique_groups > 1:  # Kruskal-Wallis requires at least two groups
                    try:
                        groups = [valid_rows[cont][valid_rows[cat] == level] for level in valid_rows[cat].unique()]
                        h_stat, p_value = kruskal(*groups)
                        results.append({
                            'Categorical Variable': cat,
                            'Continuous Variable': cont,
                            'Test Type': 'Kruskal-Wallis',
                            'H-Statistic': h_stat,
                            'P-Value': p_value
                        })
                    except Exception as e:
                        results.append({
                            'Categorical Variable': cat,
                            'Continuous Variable': cont,
                            'Test Type': 'Error',
                            'H-Statistic': None,
                            'P-Value': None,
                            'Error': str(e)
                        })

    # Return the results as a DataFrame for easy viewing
    return pd.DataFrame(results)

def cramers_v_matrix(df):
    categorical_columns = df.columns
    n = len(categorical_columns)
    cramers_v_values = np.zeros((n, n))

    # Compute Cramér's V for all pairs of columns
    for i in range(n):
        for j in range(n):
            if i == j:
                cramers_v_values[i, j] = 1.0  # Diagonal elements
            else:
                contingency_table = pd.crosstab(df.iloc[:, i], df.iloc[:, j])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n_obs = contingency_table.sum().sum()
                min_dim = min(contingency_table.shape) - 1
                cramers_v_values[i, j] = np.sqrt(chi2 / (n_obs * min_dim))

    # Convert the results into a DataFrame for easier visualization
    cramers_v_df = pd.DataFrame(cramers_v_values, index=categorical_columns, columns=categorical_columns)
    return cramers_v_df

def cramers_v_heatmap(df):
    cramers_v_df = cramers_v_matrix(df)
    mask = np.triu(np.ones_like(cramers_v_df, dtype=bool))
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        cramers_v_df,
        mask=mask,  # Mask the upper triangle
        annot=True,
        cmap="coolwarm",
        fmt=".2f",  # Format the annotations with 2 decimal places
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}  # Shrink color bar for better fit
    )
    plt.title("Cramér's V Heatmap")
    plt.show()
    
def eta_squared(anova_result):
    """
    Calculate Eta-Squared from ANOVA results.
    """
    ss_between = anova_result["sum_sq"][0]  # Sum of squares between groups
    ss_total = anova_result["sum_sq"].sum()  # Total sum of squares
    return ss_between / ss_total

def continuous_vs_categorical_heatmap(df, continuous_cols, categorical_cols):
    eta_squared_matrix = np.zeros((len(continuous_cols), len(categorical_cols)))

    for i, cont_col in enumerate(continuous_cols):
        for j, cat_col in enumerate(categorical_cols):
            # Drop rows with missing values in the current continuous or categorical column
            filtered_df = df[[cont_col, cat_col]].dropna()  # Filter out missing values

            if filtered_df.empty:  # Skip if there's no data after dropping NaNs
                eta_squared_matrix[i, j] = np.nan
                continue

            # Perform one-way ANOVA for continuous vs categorical
            formula = f"{cont_col} ~ C({cat_col})"
            try:
                model = ols(formula, data=filtered_df).fit()
                anova_result = anova_lm(model)
                # Calculate Eta-Squared and store it in the matrix
                eta_squared_matrix[i, j] = eta_squared(anova_result)
            except Exception as e:
                eta_squared_matrix[i, j] = np.nan  # Handle edge cases gracefully

    # Convert to DataFrame for heatmap
    eta_squared_df = pd.DataFrame(eta_squared_matrix, index=continuous_cols, columns=categorical_cols)

    # Plot the heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        eta_squared_df,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Eta-Squared Heatmap (Continuous vs Categorical)")
    plt.show()

def plot_histogram_boxplot(df, column, bins=40):
    # Drop missing values for the selected column
    data = df[column].dropna()
    
    # Calculate statistics
    mean = data.mean()
    median = data.median()
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantile_values = data.quantile(percentiles)
    
    # Create a figure with two subplots
    plt.figure(figsize=(12, 6))

    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, bins=bins, color='skyblue', edgecolor='black')
    
    # Add percentile lines
    for percentile, value in zip(percentiles, quantile_values):
        plt.axvline(value, linestyle='dashed', linewidth=2, color='grey', alpha=0.6, label=f'{int(percentile*100)}th Percentile: {value:.2f}')
    
    # Add title, labels, and legend
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    
    # Add horizontal grid lines for better readability
    y_ticks = plt.gca().get_yticks()  # Get current y-axis tick values
    plt.hlines(y=y_ticks, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1], 
               colors='gray', linestyles='--', linewidth=0.5, alpha=0.7)

    # Subplot 2: Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=data, color='skyblue')
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    

def plot_distributions(df, cols, color="lightblue"):

    num_cols = len(cols)
    fig, axes = plt.subplots(num_cols, 1, figsize=(8, num_cols * 4), constrained_layout=True)
    sns.set(style="white")  # Use a clean style without gridlines

    # Ensure axes is iterable for single subplot case
    if num_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, cols):
        if col in df.columns:
            sns.histplot(
                data=df,
                x=col,
                bins=20,  # Default bin size
                color=color,  # Customizable color
                edgecolor="black",
                alpha=0.9,  # Slight transparency for modern look
                ax=ax
            )
            # Add individual titles and labels
            ax.set_title(f'{col}', fontsize=14, color="black")
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
        else:
            # Handle missing columns
            ax.text(0.5, 0.5, f'{col} not found in DataFrame', fontsize=12, ha='center', va='center')
            ax.set_title(f'{col} (Missing)', fontsize=14, color="red")
            ax.axis("off")  # Hide axes for missing columns
    plt.show()
    
def plot_feature_relationships(data, target, figsize=(12, 6)):

    for col in data.columns:
        if col == target or col == 'schoolID' or col == 'studentID':
            continue  # Skip these columns
        
        plt.figure(figsize=figsize)
        
        if pd.api.types.is_numeric_dtype(data[col]):
            # Continuous feature: Scatter Plot
            sns.scatterplot(x=data[col], y=data[target], alpha=0.6)
            plt.title(f"Scatter Plot: {col} vs {target}", fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel(target, fontsize=12)
        
        elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object':
            # Categorical feature: Box Plot
            sns.boxplot(x=data[col], y=data[target])
            plt.title(f"Box Plot: {col} vs {target}", fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel(target, fontsize=12)
            plt.xticks(rotation=45)
        
        else:
            # Unknown data type: Show warning and skip
            print(f"Skipping feature '{col}': Unsupported data type ({data[col].dtype})")
        
        plt.tight_layout()
        plt.show()