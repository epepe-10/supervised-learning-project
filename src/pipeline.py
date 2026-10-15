"""Build the full Pipeline: raw data -> new features -> preprocessing -> model.

(S5) "A deployable model is a fitted Pipeline." The Pipeline holds every step
from the raw census row to the predicted probability, so the exact same
transformations are used in cross-validation, on the test set and in production.
"""

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.features import add_features

MODEL_NAMES = ["logistic_regression", "random_forest", "lightgbm"]


def build_preprocessor(use_new_features=True):
    """ColumnTransformer: one preprocessing recipe per type of column (S5)."""
    numeric_columns = config.NUMERIC_FEATURES
    if use_new_features:
        numeric_columns = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES

    # Numerical columns: fill missing values with the TRAINING median, then
    # standardize: x_scaled = (x - mean_train) / std_train.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical columns: fill missing values with the most frequent TRAINING
    # category, then one-hot encode ("categories are not measurements", S5).
    # handle_unknown="ignore": a category never seen in training becomes all
    # zeros instead of crashing the prediction.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Columns not listed here ("education", "fnlwgt") are dropped (remainder="drop").
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
    ], remainder="drop")
    return preprocessor


def build_model(model_name, params=None):
    """Create one of the three models, with optional hyperparameters.

    params is a dictionary of hyperparameters, e.g. {"C": 0.1}. When it is
    None, the model uses its default values (our untuned baseline).
    """
    params = params or {}

    if model_name == "logistic_regression":
        # (S1) Simple, interpretable, probabilistic: a strong baseline.
        # (S2) Regularization: C = 1 / lambda (small C = strong penalty) and
        #      l1_ratio picks the penalty: 0 = Ridge (L2), 1 = Lasso (L1),
        #      in between = Elastic Net. "saga" is the solver supporting all three.
        return LogisticRegression(solver="saga", max_iter=2000,
                                  random_state=config.RANDOM_STATE, **params)

    if model_name == "random_forest":
        # (S4) Many decorrelated trees (bootstrap + random features), averaged.
        # n_jobs=-1 trains the trees in parallel on all CPU cores.
        return RandomForestClassifier(n_jobs=-1, random_state=config.RANDOM_STATE, **params)

    if model_name == "lightgbm":
        # (S4) Gradient boosting: trees added one by one, each correcting the
        # current ensemble. (S5) LightGBM is the course's working model.
        # verbose=-1 hides LightGBM's training logs.
        return LGBMClassifier(random_state=config.RANDOM_STATE, verbose=-1, **params)

    raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_NAMES}")


def build_pipeline(model_name, params=None, use_new_features=True):
    """Full Pipeline. Nothing is fitted here: we only define the steps (S5).

    use_new_features=False builds the same Pipeline without our engineered
    features, to measure what feature engineering buys us (baseline.py).
    """
    steps = []
    # 1. Add the engineered features (stateless, see features.py).
    if use_new_features:
        steps.append(("features", FunctionTransformer(add_features)))
    # 2. Impute, scale and encode, learned on the training rows only.
    steps.append(("preprocessor", build_preprocessor(use_new_features)))
    # 3. The classifier.
    steps.append(("model", build_model(model_name, params)))
    return Pipeline(steps)
