import polars as pl

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from eval_utils import print_report, plot_roc_curves, plot_pr_curves

from sklearn.inspection import permutation_importance

import pandas as pd
import matplotlib.pyplot as plt


# pyright: reportUnknownMemberType=none

def eval():
    df = pl.read_parquet("data/clean/cleaned_data.parquet")

    df = df.filter(pl.col("OutcomeType") != "FOSTER") # Remove foster outcomes, since they are temporary


    # Remove target values
    X = df.select([
        "IntakeType",
        "AgeDays",
        "IntakeMonth",
        "PrimaryBreed",
        "PrimaryColor",
        "SecondaryColor",
        "IntakeIsNursing",
        "IntakeMedicalIssueIndex",
    ])

    y = (df.select("OutcomeType") == "ADOPTION").cast(pl.Int8).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
   
    predictions = {}
    probabilities = {}
    models = [random_forest(), dummy_model(), gradient_boosting()]

    for model, name in models:
        _ = model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)

    rf_importances(models[0][0], models[0][1], X_test, y_test)

    # Count positive and negative samples
    unique, counts = pl.Series(y_test).value_counts().sort("count").to_numpy().tolist()
    print(f"Test set class distribution: {dict(zip(unique, counts))}")


    print("\n=== Binary Classifier Comparison (enjoyment > ()) ===")
    for model, name in models:
        print_report(name, y_test, predictions[name], probabilities[name])

    plot_roc_curves(y_test, probabilities, out_html="roc_curve.html")
    plot_pr_curves(y_test, probabilities, out_html="pr_curve.html")

    print("\nSaved interactive plots:")
    print(" - roc_curve.html")
    print(" - pr_curve.html")

    print("See rf_permutation_importance.png for feature importances.")

def rf_importances(model, name, X_test, y_test):
    X_test = X_test.to_pandas()
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=2, scoring="f1"
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X_test.columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances for Adoption Classification")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in F1 score")
    ax.figure.tight_layout()
        
    # save to file
    plt.savefig("rf_permutation_importance.png")

def random_forest():

    numeric_features = [
        "AgeDays",
        "IntakeMonth",
        "IntakeMedicalIssueIndex",
    ]

    categorical_features = [
        "IntakeType",
        "PrimaryBreed",
        "PrimaryColor",
        "SecondaryColor",
        "IntakeIsNursing",
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both types of preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Full pipeline including the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    return (model, "Random Forest Classifier")


def gradient_boosting():

    numeric_features = [
        "AgeDays",
        "IntakeMonth",
        "IntakeMedicalIssueIndex",
    ]

    categorical_features = [
        "IntakeType",
        "PrimaryBreed",
        "PrimaryColor",
        "SecondaryColor",
        "IntakeIsNursing",
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both types of preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Full pipeline including the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier())
    ])
    return (model, "Gradient Boosting Classifier")


def dummy_model():
    model = DummyClassifier(strategy="most_frequent")
    return (model, "Dummy Classifier")

if __name__ == "__main__":
    eval()
