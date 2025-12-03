import polars as pl

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier


from sklearn.metrics import classification_report

from eval_utils import print_report, plot_roc_curves, plot_pr_curves

from sklearn.inspection import permutation_importance

import pandas as pd
import matplotlib.pyplot as plt

import shap
import numpy as np
import joblib
import os
import joblib


# pyright: reportUnknownMemberType=none

def get_data():
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
        "SpayedNeutered",
        "Sex"
    ]).to_pandas()

    y = (df.select("OutcomeType") == "ADOPTION").cast(pl.Int8).to_numpy().ravel()
    return X, y

def eval():
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
   
    predictions = {}
    probabilities = {}
    models = [random_forest(), dummy_model(), logistic_regression()]

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

    cm = confusion_matrix(y_test, predictions[models[0][1]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title("Confusion Matrix of Adoption Classification")
    fig.tight_layout()

    output_path = "random_forest_confusion_matrix.png"
    fig.savefig(output_path)
    plt.close(fig)

    pipeline = models[0][0]

    # Save the trained classification pipeline to disk for interactive use
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    clf_path = os.path.join(models_dir, "rf_classifier_pipeline.joblib")
    joblib.dump(pipeline, clf_path)
    print(f"Saved classification pipeline to: {clf_path}")

    X_preprocessed = np.asarray(pipeline.named_steps["preprocessor"].transform(X), dtype=np.float64)

    feature_names = (
        pipeline.named_steps["preprocessor"]
                .get_feature_names_out()
    )
    rf = pipeline.named_steps["classifier"]

    # subsample for SHAP to keep it fast
    max_shap_samples = 1000
    print(X_preprocessed.shape)
    if X_preprocessed.shape[0] > max_shap_samples:
        idx = np.random.RandomState(0).choice(
            X_preprocessed.shape[0], size=max_shap_samples, replace=False
        )
        X_shap = X_preprocessed[idx]
    else:
        X_shap = X_preprocessed
    print("Using SHAP input shape:", X_shap.shape)

    print("Starting SHAP value computation...")
    explainer = shap.TreeExplainer(model=rf, feature_names=feature_names)

    # New-style SHAP API: explainer(...) returns a shap.Explanation
    shap_values = explainer(X_shap)

    # For binary classification, index class 1
    fig = plt.figure(figsize=(15, 10))
    shap.plots.violin(
        shap_values[:, :, 1],
        features=X_shap,
        plot_type="layered_violin",
        show=False,
    )
    fig.savefig("shap_summary_plot.png", bbox_inches="tight")
    plt.close(fig)

    print("Saved SHAP summary plot: shap_summary_plot.png")

    print("Point #1")
    # Clear plt
    plt.clf()

    for idx in range(10):
        plt.clf()
        shap.plots.waterfall(shap_values[idx, :, 1], show=False)
        plt.title(f"SHAP Waterfall for Adoption Chance (Actual: {"ADOPTED" if y_test[idx] == 1 else "NOT ADOPTED"})")
        plt.savefig(f"shap_force_plot_{idx}.png", bbox_inches="tight")
        plt.close()


def rf_importances(model, name, X_test, y_test):
    X_test = X_test
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
        "SpayedNeutered",
        "PrimaryBreed",
        "PrimaryColor",
        "SecondaryColor",
        "IntakeIsNursing",
        "Sex",
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
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

def random_forest_no_external():

    numeric_features = [
        "AgeDays",
        "IntakeMonth",
        "IntakeMedicalIssueIndex",
    ]
    categorical_features = [
        "IntakeType",
        # "PrimaryBreed",
        # "PrimaryColor",
        # "SecondaryColor",
        # "IntakeIsNursing",
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
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
    return (model, "Random Forest Classifier without External Features")


def balanced_bagging_classifier():

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
        ('classifier', BalancedBaggingClassifier())
    ])
    return (model, "Balanced Bagging Classifier")


def do_grid_search():
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Get your pipeline and name
    model, name = random_forest()

    # Define a **small, quick** grid
    param_grid = {
        "classifier__n_estimators": [100, 200, 300, 500, 800],
        "classifier__max_depth": [None, 5, 10, 20, 30, 50],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__bootstrap": [True, False],
        "classifier__max_features": ["auto", "sqrt", "log2"],
    }

    # GridSearch setup
    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=3,
        n_iter=100,
        scoring="f1",
        verbose=10,
    )

    # Fit
    grid_search.fit(X_train, y_train)

    print("Best Params:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


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


def logistic_regression():

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
        ('classifier', LogisticRegression())
    ])
    return (model, "Logistic Regression")



def dummy_model():
    model = DummyClassifier(strategy="most_frequent")
    return (model, "Dummy Classifier")

if __name__ == "__main__":
    eval()
    do_grid_search()
