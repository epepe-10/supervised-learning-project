"""Load, clean and split the Adult (Census Income) dataset.

Usage from another script:
    from src.data import get_dev_test_data
    X_dev, X_test, y_dev, y_test = get_dev_test_data()
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src import config


def load_raw_data():
    """Download Adult from OpenML and return the raw features X and labels y."""
    # The first call downloads the data; later calls read the local cache in data/.
    adult = fetch_openml(name="adult", version=2, as_frame=True, data_home=config.DATA_DIR)
    return adult.data, adult.target


def clean_data(X, y):
    """Remove duplicates, use plain text for categories and turn the target into 0/1."""
    # Put features and target together so duplicates are checked on full rows.
    df = X.copy()
    df[config.TARGET_NAME] = y

    # (S1) Exact duplicate rows add no information and can end up in both the
    # training and validation folds. We drop them and report how many (see eda.py).
    df = df.drop_duplicates().reset_index(drop=True)

    # OpenML returns categories as the pandas "category" type. We convert them to
    # plain text so the imputer and the encoder treat every column the same way.
    for column in df.columns:
        if isinstance(df[column].dtype, pd.CategoricalDtype):
            df[column] = df[column].astype("object")

    # Binary target: 1 = income above 50K (positive, minority class), 0 = otherwise.
    y_clean = (df.pop(config.TARGET_NAME) == config.POSITIVE_LABEL).astype(int)
    return df, y_clean


def split_data(X, y):
    """Split into an 80% development set and a 20% held-out test set."""
    # (S1) stratify=y keeps the same share of ">50K" in both sets.
    # (S1) The test set is NOT another validation set: it stays untouched until
    #      the very end, while we choose preprocessing, models, hyperparameters
    #      and thresholds with cross-validation on the development set.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    return X_dev, X_test, y_dev, y_test


def get_dev_test_data():
    """Shortcut used by the other scripts: load -> clean -> split."""
    X_raw, y_raw = load_raw_data()
    X, y = clean_data(X_raw, y_raw)
    return split_data(X, y)
