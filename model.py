import polars as pl

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from eval_utils import print_report, plot_roc_curves, plot_pr_curves

from sklearn.inspection import permutation_importance

import pandas as pd
import matplotlib.pyplot as plt


# pyright: reportUnknownMemberType=none

def eval():
    df = pl.read_parquet("data/clean/cleaned_data.parquet")


    # Remove target values
    X = df.select([
        "IntakeType",
        "AgeDays",
        "IntakeMonth",
        "Sex",
        "PrimaryBreed",
        "PrimaryColor",
        "SecondaryColor",
        "IntakeIsNursing",
        "IntakeMedicalIssueIndex",
        "SpayedNeutered",
    ])

    y = (df.select("OutcomeType") == "ADOPTION").cast(pl.Int8).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
   
    predictions = {}
    probabilities = {}
    models = [random_forest(), dummy_model(), external_chars_rf()]

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

def rf_importances(model, name, X_test, y_test):
    X_test = X_test.to_pandas()
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=2, scoring="f1"
    )

    feature_names = list(X_test.columns)  # original columns
    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title(f"Feature importances using permutation on {name}")
    ax.set_ylabel("Mean F1 decrease")
    fig.tight_layout()
    
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
        "Sex",
        "PrimaryBreed",
        "PrimaryColor",
        "SecondaryColor",
        "IntakeIsNursing",
        "SpayedNeutered",
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


def external_chars_rf():

    numeric_features = [
    ]

    categorical_features = [
        "PrimaryBreed",
        "PrimaryColor",
        "SecondaryColor",
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
    return (model, "Random Forest Classifier with external characteristics")


def dummy_model():
    model = DummyClassifier(strategy="most_frequent")
    return (model, "Dummy Classifier")

if __name__ == "__main__":
    eval()
