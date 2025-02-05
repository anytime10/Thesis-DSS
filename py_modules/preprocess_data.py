import pandas as pd
import numpy as np
import re
from sklearn import set_config 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer

set_config(transform_output="pandas")

def preprocess_data(
    train,
    test,
    columns_to_transform,
    columns_to_scale_and_transform,
    only_to_impute,
    one_hot_cols_fr,
    one_hot_cols,
    ordinal_columns,
    dummy_cols,
    dropping_category
):
    # Handle missing values for dummy columns
    for col in dummy_cols:
        train[col] = train[col].apply(lambda x: 0 if pd.isna(x) else 1)
        test[col] = test[col].apply(lambda x: 0 if pd.isna(x) else 1)

    # Define the category orders for ordinal encoding
    clsize_categories = ['15 students or fewer', '16-20 students', '21-25 students', '26-30 students']
    math_grouping_categories = ['Not for any classes', 'For some classes', 'For all classes']
    municipality_size = [
        'A small town (3 000 to about 15 000 people)',
        'A town (15 000 to about 100 000 people)',
        'A city (100 000 to about 1 000 000 people)'
    ]
    competition = [
        'There are no other schools in this area that compete for our students.',
        'There is one other school in this area that competes for our students.',
        'There are two or more other schools in this area that compete for our students.'
    ]
    edushort = ['Low', 'Middle', 'High']

    # Define pipelines for different preprocessing tasks
    cat_pipeline_1 = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot', OneHotEncoder(drop=dropping_category, sparse_output=False, handle_unknown='ignore'))
    ])
    cat_pipeline_2 = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('one_hot', OneHotEncoder(drop=dropping_category, sparse_output=False, handle_unknown='ignore'))
    ])
    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_encoder', OrdinalEncoder(categories=[
            clsize_categories,
            math_grouping_categories,
            competition,
            municipality_size,
            edushort
        ]))
    ])
    num_pipeline1 = Pipeline([
        ('knn_imputer', KNNImputer(n_neighbors=5)),
        ('power_transformer', PowerTransformer(method='yeo-johnson', standardize=False))
    ])
    num_pipeline2 = Pipeline([
        ('knn_imputer', KNNImputer(n_neighbors=5)),
        ('power_transformer', PowerTransformer(method='yeo-johnson', standardize=True))
    ])
    impute_only_num = Pipeline([
        ('imputer_num', KNNImputer(n_neighbors=5))
    ])
    impute_only_ord = Pipeline([
        ('imputer_ord', SimpleImputer(strategy='most_frequent'))
    ])

    # Combine pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num1', num_pipeline1, columns_to_transform),
        ('num2', num_pipeline2, columns_to_scale_and_transform),
        ('cat_1', cat_pipeline_1, one_hot_cols_fr),
        ('cat_2', cat_pipeline_2, one_hot_cols),
        ('ord', ord_pipeline, ordinal_columns),
        ('impute_num', impute_only_num, only_to_impute),
        ('impute_ord', impute_only_ord, ['SCHAUTO_ORDINAL'])
    ], remainder="passthrough")

    # Apply preprocessing
    X_train_preprocessed = preprocessor.fit_transform(train)
    X_test_preprocessed = preprocessor.transform(test)

    # Rename columns (if applicable)
    X_train_preprocessed.columns = [re.sub(r'^.*__', '', col) for col in X_train_preprocessed.columns]
    X_test_preprocessed.columns = [re.sub(r'^.*__', '', col) for col in X_test_preprocessed.columns]

    return X_train_preprocessed, X_test_preprocessed








