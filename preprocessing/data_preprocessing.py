# preprocessing/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df, fit_preprocessor=None):
    """
    Prétraitement des données : imputation, scaling, encodage.
    - Si fit_preprocessor est None, fit+transform (train)
    - Si fit_preprocessor est fourni, transform uniquement (test)
    Retourne : X_processed (numpy array), y (Series ou None), preprocessor
    """
    # Séparation X / y si la colonne 'y' existe
    if 'y' in df.columns:
        y = df['y']
        X = df.drop('y', axis=1)
    else:
        y = None
        X = df.copy()

    # Séparation colonnes numériques / catégorielles
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    transformers = []
    if num_cols:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, num_cols))
    if cat_cols:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers)

    if fit_preprocessor is None:
        X_processed = preprocessor.fit_transform(X)
        return X_processed, y, preprocessor
    else:
        X_processed = fit_preprocessor.transform(X)
        return X_processed, y