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
    models = [random_forest(), dummy_model()]

    for model, name in models:
        _ = model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)

    
    rf, name = random_forest()
    _ = rf.fit(X_train, y_train)
    feature_names = rf.named_steps['preprocessor'].get_feature_names_out()
    importances = rf.named_steps['classifier'].feature_importances_
    feature_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("=== Random Forest Feature Importances ===")
    for feature, importance in feature_importances[:10]:
        print(f"{feature}: {importance:.4f}")


    print("\n=== Binary Classifier Comparison (enjoyment > ()) ===")
    for model, name in models:
        print_report(name, y_test, predictions[name], probabilities[name])

    plot_roc_curves(y_test, probabilities, out_html="roc_curve.html")
    plot_pr_curves(y_test, probabilities, out_html="pr_curve.html")

    print("\nSaved interactive plots:")
    print(" - roc_curve.html")
    print(" - pr_curve.html")

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
    return (model, "Logistic Regression")

def dummy_model():
    model = DummyClassifier(strategy="most_frequent")
    return (model, "Dummy Classifier")

if __name__ == "__main__":
    eval()
