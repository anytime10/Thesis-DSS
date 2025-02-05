from sklearn.model_selection import GroupShuffleSplit
    
def split_and_save_data(df, target_col='PV1MATH', group_col='schoolID', test_size=0.2, random_state=42, output_dir="."):
    # Separate target and groups
    y = df[target_col].copy()
    groups = df[group_col].copy()

    # Drop target, group_col, and IDs to create features
    X = df.drop([target_col, group_col, "studentID"], axis=1, errors='ignore')

    # Initialize GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    # Split data into train and test sets
    X_train = X.iloc[train_idx].copy().reset_index(drop=True)
    X_test = X.iloc[test_idx].copy().reset_index(drop=True)
    y_train = y.iloc[train_idx].copy().reset_index(drop=True)
    y_test = y.iloc[test_idx].copy().reset_index(drop=True)

    # Extract clusters for train and test sets
    clusters_train = df['PROGN'].iloc[train_idx].copy().reset_index(drop=True)
    clusters_test = df['PROGN'].iloc[test_idx].copy().reset_index(drop=True)

    # Save data to CSV
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False, header=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False, header=False)

    # Return these objects
    return {
        "groups": groups,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "clusters_train": clusters_train,  # Add clusters for train
        "clusters_test": clusters_test,    # Add clusters for test
        "gss": gss,
    }


