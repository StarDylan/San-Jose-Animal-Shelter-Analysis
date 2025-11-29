"""
Simple interactive CLI to create a cat and predict:
 - probability of adoption (classification model)
 - predicted days until adoption (regression model)

This script expects two saved pipelines produced by `model.py` and `regression.py`:
 - models/rf_classifier_pipeline.joblib
 - models/rf_regression_pipeline.joblib

If those files are missing, run `python model.py` and `python regression.py` to train and save them.
"""

from __future__ import annotations

import os
import joblib
import pandas as pd
import polars as pl
import numpy as np

MODELS_DIR = "models"
CLF_PATH = os.path.join(MODELS_DIR, "rf_classifier_pipeline.joblib")
REGR_PATH = os.path.join(MODELS_DIR, "rf_regression_pipeline.joblib")


def prompt(prompt_text: str, default: str | None = None) -> str:
    if default is not None:
        return input(f"{prompt_text} [{default}]: ") or default
    return input(f"{prompt_text}: ")


def load_models():
    if not os.path.exists(CLF_PATH) or not os.path.exists(REGR_PATH):
        raise FileNotFoundError(
            "Saved pipeline(s) not found. Run `python model.py` and `python regression.py` to create them." 
        )
    clf = joblib.load(CLF_PATH)
    regr = joblib.load(REGR_PATH)
    return clf, regr


def load_reference_values(parquet_path: str = "data/clean/cleaned_data.parquet"):
    """Load unique values from the cleaned dataset to offer choices interactively."""
    if not os.path.exists(parquet_path):
        print(f"Reference data not found at {parquet_path}. Interactive choices will be limited.")
        return {}
    df = pl.read_parquet(parquet_path)

    def uniq(col: str):
        if col not in df.columns:
            return []
        # Return values ordered by frequency (most common first). Keep None if present.
        # Use pandas value_counts for convenience (handles NaN).
        s = df.select(col).to_pandas()[col]
        vc = s.value_counts(dropna=False)
        vals = []
        for v in vc.index.tolist():
            # pandas uses NaN for nulls; convert to None for display/selection
            if pd.isna(v):
                vals.append(None)
            else:
                vals.append(v)
        return vals

    # For categorical lists, order by frequency (most common first). For month keep numeric order.
    intake_type_vals = uniq("IntakeType")
    month_vals = uniq("IntakeMonth")
    primary_breed_vals = uniq("PrimaryBreed")
    primary_color_vals = uniq("PrimaryColor")
    secondary_color_vals = uniq("SecondaryColor")

    # Build ref with frequency-sorted categorical lists (None kept if present).
    def freq_list(L):
        if not L:
            return []
        # L is already frequency-ordered from uniq(); return as-is
        return L

    # For months, prefer numeric sorted order; if None present, append at end
    def month_list(L):
        nums = [int(v) for v in L if v is not None]
        nums = sorted(set(nums))
        if any(v is None for v in L):
            nums.append(None)
        return nums

    ref = {
        "IntakeType": freq_list(intake_type_vals),
        "IntakeMonth": month_list(month_vals),
        "AgeDays": df.select("AgeDays").to_series().to_numpy() if "AgeDays" in df.columns else np.array([]),
        "YoungCatAgeDays": df.select("YoungCatAgeDays").to_series().to_numpy() if "YoungCatAgeDays" in df.columns else np.array([]),
        "OldCatAgeYears": df.select("OldCatAgeYears").to_series().to_numpy() if "OldCatAgeYears" in df.columns else np.array([]),
        "PrimaryBreed": freq_list(primary_breed_vals),
        "PrimaryColor": freq_list(primary_color_vals),
        "SecondaryColor": freq_list(secondary_color_vals),
    }
    return ref


def choose_from_list(options: list, label: str, max_display: int = 40):
    """Show options with indices; allow substring search or index selection.

    Returns the original option value (not its display string). If options is empty
    the function falls back to a simple text prompt.
    """
    if not options:
        return prompt(f"{label} (no reference options available)")

    # prepare display strings while keeping original values
    entries = []  # list of (orig, display_str)
    for orig in options:
        if orig is None:
            disp = "<None>"
        else:
            disp = str(orig)
        entries.append((orig, disp))

    while True:
        print(f"\nSelect {label} — enter index number or type text to search:")
        for i, (_, disp) in enumerate(entries[:max_display]):
            print(f" {i:3d}: {disp}")
        if len(entries) > max_display:
            print(f" ... and {len(entries)-max_display} more. Type a substring to filter.")

        choice = input("Choice (index or text): ").strip()
        if choice == "":
            print("Please enter a selection.")
            continue

        # numeric index selection
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(entries):
                return entries[idx][0]
            else:
                print("Index out of range.")
                continue

        # substring search against display strings
        matches = [orig for (orig, disp) in entries if choice.lower() in disp.lower()]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Found {len(matches)} matches:")
            for i, v in enumerate(matches[:max_display]):
                # show display for the match
                disp = next(d for (o, d) in entries if o == v)
                print(f" {i:3d}: {disp}")
            subchoice = input("Enter index of match or press Enter to search again: ").strip()
            if subchoice.isdigit():
                subidx = int(subchoice)
                if 0 <= subidx < len(matches):
                    return matches[subidx]
            continue
        else:
            use_raw = input(f"No matches found. Use '{choice}' as raw value? (y/n) [n]: ").strip().lower()
            if use_raw in ("y", "yes"):
                return choice


def choose_month(month_options: list):
    months = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
    # Build display options preserving order from month_options; include None as <None>
    display_opts = []
    for m in month_options:
        if m is None:
            display_opts.append(None)
        else:
            try:
                mi = int(m)
                if mi in months:
                    display_opts.append(mi)
            except Exception:
                # skip invalid entries
                continue

    # create human-readable display strings for the chooser (include number)
    chooser_list = []
    for m in display_opts:
        if m is None:
            chooser_list.append("<None>")
        else:
            chooser_list.append(f"{m} - {months.get(int(m), str(m))}")

    sel = choose_from_list(chooser_list, "IntakeMonth")
    if sel is None:
        return None
    # sel is a display string like '6 - June' or '<None>' or possibly a raw string
    if isinstance(sel, str) and " - " in sel:
        return int(sel.split(" - ")[0])
    try:
        return int(sel)
    except Exception:
        return None


def choose_numeric_from_distribution(values: np.ndarray, label: str):
    if values is None or len(values) == 0:
        return float(prompt(f"{label} (numeric)", "0"))
    vals = np.array(values, dtype=float)
    # Remove NaN/inf values before computing percentiles
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(prompt(f"{label} (numeric)", "0"))
    percentiles = np.percentile(vals, [0, 10, 25, 50, 75, 90, 100])
    choices = [f"p{p}: {int(percentiles[i])}" for i, p in enumerate([0, 10, 25, 50, 75, 90, 100])]
    print(f"\nSelect {label} from percentiles or enter custom number:")
    for i, c in enumerate(choices):
        print(f" {i}: {c}")
    choice = input("Choice index or numeric value: ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(choices):
            # split on ':' and strip whitespace
            return float(choices[idx].split(":")[1].strip())
    try:
        return float(choice)
    except Exception:
        print("Invalid numeric input; using median.")
        return float(percentiles[3])


def make_input_row_classification(ref: dict) -> pd.DataFrame:
    print("Enter features for classification model (Adoption yes/no):")
    IntakeType = choose_from_list(ref.get("IntakeType", []), "IntakeType")
    AgeDays = choose_numeric_from_distribution(ref.get("AgeDays", np.array([])), "AgeDays")
    IntakeMonth = choose_month(ref.get("IntakeMonth", list(range(1, 13))))
    PrimaryBreed = choose_from_list(ref.get("PrimaryBreed", []), "PrimaryBreed")
    PrimaryColor = choose_from_list(ref.get("PrimaryColor", []), "PrimaryColor")
    SecondaryColor = choose_from_list(ref.get("SecondaryColor", []), "SecondaryColor")
    IntakeIsNursing = prompt("IntakeIsNursing (yes/no)", "no").lower() in ("y", "yes", "true", "1")
    IntakeMedicalIssueIndex = float(prompt("IntakeMedicalIssueIndex (numeric)", "0"))

    row = {
        "IntakeType": IntakeType,
        "AgeDays": AgeDays,
        "IntakeMonth": IntakeMonth,
        "PrimaryBreed": PrimaryBreed,
        "PrimaryColor": PrimaryColor,
        "SecondaryColor": SecondaryColor,
        "IntakeIsNursing": IntakeIsNursing,
        "IntakeMedicalIssueIndex": IntakeMedicalIssueIndex,
    }
    return pd.DataFrame([row])


def make_input_row_regression(ref: dict) -> pd.DataFrame:
    """Regression input: prefer `AgeDays` if present (single numeric), otherwise fall back
    to the dataset's `YoungCatAgeDays`/`OldCatAgeYears` fields. If only AgeDays is
    provided we populate both expected regression features using AgeDays (OldCatAgeYears
    = AgeDays / 365).
    """
    print("\nEnter features for regression model (Time until adoption):")
    IntakeType = choose_from_list(ref.get("IntakeType", []), "IntakeType")

    # Prefer AgeDays if available (single numeric input)
    if "AgeDays" in ref and ref.get("AgeDays") is not None and len(ref.get("AgeDays")) > 0:
        AgeDays = choose_numeric_from_distribution(ref.get("AgeDays", np.array([])), "AgeDays")
        # populate the regression features expected by the model
        YoungCatAgeDays = float(AgeDays)
        OldCatAgeYears = float(AgeDays) / 365.0
    else:
        YoungCatAgeDays = choose_numeric_from_distribution(ref.get("YoungCatAgeDays", np.array([])), "YoungCatAgeDays")
        OldCatAgeYears = choose_numeric_from_distribution(ref.get("OldCatAgeYears", np.array([])), "OldCatAgeYears")

    IntakeMonth = choose_month(ref.get("IntakeMonth", list(range(1, 13))))
    PrimaryBreed = choose_from_list(ref.get("PrimaryBreed", []), "PrimaryBreed")
    PrimaryColor = choose_from_list(ref.get("PrimaryColor", []), "PrimaryColor")
    SecondaryColor = choose_from_list(ref.get("SecondaryColor", []), "SecondaryColor")
    IntakeIsNursing = prompt("IntakeIsNursing (yes/no)", "no").lower() in ("y", "yes", "true", "1")
    IntakeMedicalIssueIndex = float(prompt("IntakeMedicalIssueIndex (numeric)", "0"))

    row = {
        "IntakeType": IntakeType,
        "YoungCatAgeDays": YoungCatAgeDays,
        "OldCatAgeYears": OldCatAgeYears,
        "IntakeMonth": IntakeMonth,
        "PrimaryBreed": PrimaryBreed,
        "PrimaryColor": PrimaryColor,
        "SecondaryColor": SecondaryColor,
        "IntakeIsNursing": IntakeIsNursing,
        "IntakeMedicalIssueIndex": IntakeMedicalIssueIndex,
    }
    return pd.DataFrame([row])


def main():
    print("Interactive cat adoption predictor")
    clf, regr = load_models()
    ref = load_reference_values()

    while True:
        X_clf = make_input_row_classification(ref)
        X_regr = make_input_row_regression(ref)

        # Classification: probability and prediction
        try:
            proba = clf.predict_proba(X_clf)[0]
            pred = clf.predict(X_clf)[0]
            print(f"\nAdoption probability (class=0/1): {proba}")
            print(f"Predicted adoption (label): {pred}")
        except Exception as e:
            print("Could not run classification prediction:", e)

        # Regression: predicted days until adoption
        try:
            pred_days = regr.predict(X_regr)[0]
            print(f"Predicted time until adoption (days): {pred_days:.1f}")
        except Exception as e:
            print("Could not run regression prediction:", e)

        again = prompt("Predict another cat? (y/n)", "y").lower()
        if again not in ("y", "yes"):
            break

    print("Goodbye")


if __name__ == "__main__":
    main()
