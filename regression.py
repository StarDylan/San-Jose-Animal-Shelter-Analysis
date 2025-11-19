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

    y = (df.select("TimeInShelterDays")).cast(pl.Int64).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
   
    predictions = {}
    probabilities = {}
    models = [random_forest()]

    for model, name in models:
        print(f"Evaluating model: {name}")
        _ = model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

        # Output MSE + Explained variance
        mse = mean_squared_error(y_test, predictions[name])
        evs = explained_variance_score(y_test, predictions[name])
        print(f"{name} - MSE: {mse}, Explained Variance: {evs}")



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
        ('classifier', RandomForestRegressor())
    ])
    return (model, "Random Forest Classifier")

if __name__ == "__main__":
    eval()