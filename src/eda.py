"""Exploratory Data Analysis (EDA) for the Adult dataset.

Run with:  python -m src.eda

(S1) "The goal of EDA is not to 'make plots'. It is to discover which
assumptions about the dataset are safe enough to build on."

The script prints its findings in the terminal and saves figures to
outputs/figures/. The interpretations go in the report.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.data import clean_data, load_raw_data, split_data


def print_title(title):
    """Print a visible section title in the terminal."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_figure(fig, name):
    """Save a matplotlib figure in outputs/figures/ and close it."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


# ===========================================================================
# Part 1 - Data integrity (on the FULL raw dataset, target not used)
# ===========================================================================
def check_integrity(X_raw, y_raw):
    """Shape, types, missing values and duplicates."""
    print_title("1. First inspection")
    print(f"Rows: {X_raw.shape[0]}   Columns: {X_raw.shape[1]}")

    # (S1) "The schema is more informative than the dtype": we state the
    # *meaning* of each column, not only how it is stored.
    schema = pd.DataFrame({
        "storage dtype": X_raw.dtypes.astype(str),
        "unique values": X_raw.nunique(),
    })
    schema["semantic type"] = "categorical"
    schema.loc[config.NUMERIC_FEATURES + ["fnlwgt"], "semantic type"] = "numerical"
    schema.loc["education-num", "semantic type"] = "ordered discrete"
    print(schema)

    print_title("2. Missing values")
    missing = X_raw.isna().sum()
    missing = missing[missing > 0]
    print(pd.DataFrame({"count": missing, "share (%)": (100 * missing / len(X_raw)).round(2)}))
    print("-> Only categorical columns have missing values: they will be imputed with")
    print("   the most frequent category INSIDE the Pipeline (S5), never before the split.")

    print_title("3. Duplicates")
    full_rows = X_raw.copy()
    full_rows["target"] = y_raw
    n_duplicates = full_rows.duplicated().sum()
    print(f"Exact duplicate rows: {n_duplicates} ({100 * n_duplicates / len(full_rows):.2f}%)")
    print("-> Dropped in data.clean_data().")

    print_title("4. Redundant column: education vs education-num")
    # If each education label maps to exactly one number, the two columns are
    # the same information written twice.
    numbers_per_label = X_raw.groupby("education", observed=True)["education-num"].nunique()
    print(f"Max number of education-num values per education label: {numbers_per_label.max()}")
    print("-> One-to-one mapping: we keep only 'education-num' (ordered) and drop 'education'.")


# ===========================================================================
# Part 2 - Target and predictors (on the DEVELOPMENT set only)
# ===========================================================================
# Looking at how features relate to the target is already a modelling choice,
# so we do it on the development set only: the test set must stay unseen.

def explore_target(y_dev):
    """Class balance and the accuracy of a model that always predicts the majority."""
    print_title("5. Target: class balance (development set)")
    counts = y_dev.value_counts().sort_index()
    prevalence = y_dev.mean()
    print(f"<=50K (0): {counts[0]}   >50K (1): {counts[1]}")
    print(f"Prevalence of the positive class: {100 * prevalence:.1f}%")
    # (S6) "The baseline matters when interpreting accuracy and precision."
    print(f"Accuracy of 'always predict <=50K': {100 * (1 - prevalence):.1f}%")
    print("-> Accuracy alone is misleading here: we will report ROC-AUC, Average")
    print("   Precision, precision, recall, F1 and the confusion matrix (S6).")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["<=50K (0)", ">50K (1)"], counts.values, color=["tab:blue", "tab:orange"])
    ax.set_title("Target distribution (development set)")
    ax.set_ylabel("Number of people")
    save_figure(fig, "eda_target_distribution.png")


def explore_numeric(X_dev, y_dev):
    """Distribution of each numerical feature, split by class."""
    print_title("6. Numerical features by class")
    means = X_dev[config.NUMERIC_FEATURES].groupby(y_dev).mean().round(1)
    print(means.rename(index={0: "<=50K", 1: ">50K"}).to_string())

    # Share of exact zeros: capital-gain / capital-loss are almost always 0.
    for column in ["capital-gain", "capital-loss"]:
        print(f"Share of zeros in {column}: {100 * (X_dev[column] == 0).mean():.1f}%")
    print("-> Very skewed: motivates the 'has_capital' indicator feature.")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, column in zip(axes.ravel(), config.NUMERIC_FEATURES):
        ax.hist(X_dev.loc[y_dev == 0, column], bins=30, alpha=0.6, density=True, label="<=50K")
        ax.hist(X_dev.loc[y_dev == 1, column], bins=30, alpha=0.6, density=True, label=">50K")
        ax.set_title(column)
        ax.legend()
    axes.ravel()[-1].axis("off")   # 5 features on a 2x3 grid -> hide the last empty plot
    save_figure(fig, "eda_numeric_by_class.png")

    # Age vs share of >50K: an inverted U, which a linear model cannot represent
    # on its own -> motivates the 'age_squared' feature (S3).
    rate_by_age = y_dev.groupby(X_dev["age"]).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rate_by_age.index, rate_by_age.values)
    ax.set_title("Share of >50K by age (development set)")
    ax.set_xlabel("age")
    ax.set_ylabel("share of >50K")
    save_figure(fig, "eda_rate_by_age.png")


def explore_categorical(X_dev, y_dev):
    """Share of >50K for each category of each categorical feature."""
    print_title("7. Categorical features: share of >50K per category")
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    for ax, column in zip(axes.ravel(), config.CATEGORICAL_FEATURES):
        rate = y_dev.groupby(X_dev[column]).mean().sort_values()
        # native-country has ~40 levels: show only the 15 most frequent countries.
        if column == "native-country":
            top_countries = X_dev[column].value_counts().index[:15]
            rate = rate[rate.index.isin(top_countries)]
        ax.barh(rate.index.astype(str), rate.values)
        ax.axvline(y_dev.mean(), color="red", linestyle="--", label="overall rate")
        ax.set_title(f"Share of >50K by {column}")
        ax.legend()
        print(f"{column}: {X_dev[column].nunique()} categories, "
              f"rate from {rate.min():.2f} to {rate.max():.2f}")
    axes.ravel()[-1].axis("off")
    save_figure(fig, "eda_categorical_rates.png")


def explore_sensitive(X_dev, y_dev):
    """Base rates for the sensitive attributes (input for the fairness audit)."""
    print_title("8. Sensitive attributes")
    for column in config.SENSITIVE_FEATURES:
        table = pd.DataFrame({
            "share of people (%)": (100 * X_dev[column].value_counts(normalize=True)).round(1),
            "share of >50K (%)": (100 * y_dev.groupby(X_dev[column]).mean()).round(1),
        })
        print(table, "\n")
    print("-> Large gaps in base rates already exist in the data, before any model.")


def print_leakage_audit():
    """(S1) Prediction-time test: 'Could I genuinely know this value at the
    moment the prediction is made?'"""
    print_title("9. Leakage audit")
    audit = pd.DataFrame(
        [
            ["age, sex, race, native-country", "Known", "Low", "Fixed personal characteristics"],
            ["education, education-num", "Known", "Low", "Obtained before the income is earned"],
            ["workclass, occupation, hours-per-week", "Known", "Low", "Describe the current job"],
            ["marital-status, relationship", "Known", "Low", "Household situation"],
            ["capital-gain, capital-loss", "Known (same tax year)", "Medium", "Reported with income: check in production"],
            ["fnlwgt", "Known", "Not a person feature", "Survey sampling weight -> dropped"],
        ],
        columns=["Feature group", "Prediction-time availability", "Risk", "Why"],
    )
    print(audit.to_string(index=False))


def print_conclusions():
    """(S1) EDA finishes with justified decisions and explicit open assumptions."""
    print_title("10. What we know / what remains open")
    print("What we know:")
    print("  - Binary target, ~24% positives -> imbalanced, accuracy is not enough.")
    print("  - Missing values only in workclass, occupation, native-country.")
    print("  - 'education' duplicates 'education-num'; 'fnlwgt' is a survey weight.")
    print("  - One row = one person, no timestamps -> stratified split + StratifiedKFold.")
    print("What remains open:")
    print("  - Are missing occupations random, or a group of their own (e.g. not working)?")
    print("  - capital-gain/loss may be reported at the same time as income (see audit).")
    print("  - Data is from the 1994 US census: income levels and patterns may have drifted.")


def main():
    X_raw, y_raw = load_raw_data()
    check_integrity(X_raw, y_raw)

    X, y = clean_data(X_raw, y_raw)
    X_dev, X_test, y_dev, y_test = split_data(X, y)
    print(f"\nDevelopment set: {len(X_dev)} rows   Test set: {len(X_test)} rows (not explored)")

    explore_target(y_dev)
    explore_numeric(X_dev, y_dev)
    explore_categorical(X_dev, y_dev)
    explore_sensitive(X_dev, y_dev)
    print_leakage_audit()
    print_conclusions()
    print(f"\nFigures saved in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
