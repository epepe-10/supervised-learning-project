"""Feature engineering: three new features, one per idea seen in S3.

(S3) A logistic regression is linear in the log-odds: it cannot learn
thresholds, curves or interactions on its own. Feature engineering gives
it these shapes explicitly, using the three tools from the session:
    1. a square term       -> age_squared
    2. an interaction      -> education_x_hours
    3. an indicator        -> has_capital

Tree models (Random Forest, LightGBM) can learn these shapes by themselves,
so the new features matter most for the linear baseline.
"""

from src import config


def add_features(X):
    """Return a copy of X with the engineered features added.

    The function only combines columns of the SAME row: it learns nothing
    from the data (no mean, no median, no target). So it cannot leak
    information between training and validation rows, and is safe to run
    inside the Pipeline.
    """
    X = X.copy()

    # 1. Square term: the share of >50K rises with age, peaks around 50 and
    #    then falls (see eda_rate_by_age.png). With age AND age^2 the linear
    #    model can draw this inverted U instead of a straight line.
    X["age_squared"] = X["age"] ** 2

    # 2. Interaction: "an interaction allows the effect of one feature to depend
    #    on the value of another feature" (S3). Extra hours are likely worth more
    #    for highly educated people than for others.
    X["education_x_hours"] = X["education-num"] * X["hours-per-week"]

    # 3. Indicator: capital-gain and capital-loss are 0 for more than 90% of
    #    people. Owning capital at all (yes/no) is a cleaner signal than the
    #    raw, very skewed amounts.
    X["has_capital"] = ((X["capital-gain"] > 0) | (X["capital-loss"] > 0)).astype(int)

    return X


def feature_names_out(transformer, input_features):
    """Column names after add_features: the input columns, then the new ones.

    The Pipeline needs this to answer get_feature_names_out(), which gives a
    name to each column the model sees (used later for SHAP explanations).
    """
    return [*input_features, *config.ENGINEERED_FEATURES]
