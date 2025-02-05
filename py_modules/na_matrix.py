import matplotlib.pyplot as plt
import missingno as mmsno
import matplotlib.patches as mpatches

def plot_missing_data_matrix(X, st_cols, sch_cols):
    # Reorder columns for visualization
    X_reordered = X[st_cols + sch_cols]

    # Calculate missing data statistics
    total_entries = X.size
    missing_entries = X.isnull().sum().sum()
    missing_percentage = (missing_entries / total_entries) * 100
    complete_percentage = 100 - missing_percentage

    # Plot the missing data matrix
    mmsno.matrix(X_reordered)
    plt.xticks(rotation=90)
    ax = plt.gca()
    xticks = ax.get_xticklabels()

    # Set tick colors for different feature groups
    for tick in xticks:
        if tick.get_text() in st_cols:
            tick.set_color("#8c564b")  # Student-level features
        elif tick.get_text() in sch_cols:
            tick.set_color("#1f77b4")  # School-level features

    # Add legend to indicate colors
    student_patch = mpatches.Patch(color="#8c564b", label="Student-level features")
    school_patch = mpatches.Patch(color="#1f77b4", label="School-level features")
    plt.legend(handles=[student_patch, school_patch], loc="upper left")

    # Add a summary of missing and complete data percentages
    plt.figtext(
        0.5, 0.02, 
        f"Missing Data: {missing_percentage:.2f}% | Complete Data: {complete_percentage:.2f}%", 
        ha="center", fontsize=12, color="black", bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}
    )

    plt.show()


