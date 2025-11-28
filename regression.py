from numpy import ndarray


from typing import Any


import polars as pl

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, explained_variance_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from eval_utils import print_report, plot_roc_curves, plot_pr_curves

from sklearn.inspection import permutation_importance

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# pyright: reportUnknownMemberType=none

def eval():
    df = pl.read_parquet("data/clean/cleaned_data.parquet")

    # Remove 11 negative time in shelter values
    # Only consider cats that are adopted.
    df = df.filter((pl.col("OutcomeType") == "ADOPTION") &  (pl.col("TimeInShelterDays") >= 0))

    # Remove target values
    X = df.select([
        "IntakeType",
        "YoungCatAgeDays",
        "OldCatAgeYears",
        "IntakeMonth",
        "PrimaryBreed",
        "PrimaryColor",
        "SecondaryColor",
        "IntakeIsNursing",
        "IntakeMedicalIssueIndex",
    ])

    y: ndarray[tuple[int], Any] = (df.select("TimeInShelterDays")).cast(pl.Int64).to_numpy().ravel()
    print(np.unique(y))
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
   
    predictions = {}
    probabilities = {}
    models = [random_forest()]

    for model, name in models:
        print(f"Evaluating model: {name}")
        _ = model.fit(X_train, y_train)

        print("")
        predictions[name] = model.predict(X_test)

        # Output MSE + Explained variance
        mse = mean_squared_error(y_test, predictions[name])
        evs = explained_variance_score(y_test, predictions[name])
        print(f"{name} - MSE: {mse}, Explained Variance: {evs}")

    
        rf_importances(models[0][0], models[0][1], X_test, y_test)



def rf_importances(model, name, X_test, y_test):
    X_test = X_test.to_pandas()
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=2, scoring="neg_mean_squared_error"
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X_test.columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances for Time until Adoption")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Incrase in MSE")
    ax.figure.tight_layout()
        
    # save to file
    plt.savefig("rf_permutation_importance_regression.png")

def random_forest():

    numeric_features = [
        "YoungCatAgeDays",
        "OldCatAgeYears",
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
        ('classifier', RandomForestRegressor())
    ])
    return (model, "Random Forest Regressor")

if __name__ == "__main__":
    eval()
    print("Finished evaluation. See rf_permutation_importance_regression.png for feature importances.")