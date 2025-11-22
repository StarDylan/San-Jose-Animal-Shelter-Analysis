
from enum import unique
import glob
from collections.abc import Iterable
import polars as pl
import pantab
from datetime import datetime, timedelta

# pyright: reportUnknownMemberType=none


# What is REHAB?
# Time in shelter with IntakeReason == "IP ADOPT"


import polars as pl

# --- 1) Helpers ---------------------------------------------------------------

def collapse_colors(col: pl.Expr) -> pl.Expr:
    c = (
        col.str.to_uppercase()
          .str.replace_all(r"[^\w\s/-]", "")   # drop odd chars like backslashes
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
    )
    # Normalize some aliases first
    c = (
        c.str.replace_all(r"\bBLUE\b", "GRAY")
         .str.replace_all(r"\bYELLOW\b", "ORANGE")
         .str.replace_all(r"\bBRN\b", "BROWN")
         .str.replace_all(r"\bORG\b", "ORANGE")
    )

    # Rule-based buckets
    return (
        pl.when(c.str.contains(r"\bTABBY"))      .then(pl.lit("TABBY"))
         .when(c.str.contains(r"\bTORBI"))       .then(pl.lit("TORBIE"))
         .when(c.str.contains(r"\bCALICO"))      .then(pl.lit("CALICO"))
         .when(c.str.contains(r"\bTORTIE"))      .then(pl.lit("TORTIE"))
         .when(c.str.contains(r"\bPT[- ]"))      .then(pl.lit("POINT"))   # PT-LILAC, PT-SEAL, etc.
         .when(c.str.contains(r"\bBUFF|CREAM"))  .then(pl.lit("CREAM/BUFF"))
         .when(c.str.contains(r"\bGRAY"))        .then(pl.lit("GRAY"))
         .when(c.str.contains(r"\bBLACK"))       .then(pl.lit("BLACK"))
         .when(c.str.contains(r"\bWHITE"))       .then(pl.lit("WHITE"))
         .when(c.str.contains(r"\bORANGE"))      .then(pl.lit("ORANGE"))
         .otherwise(pl.lit("OTHER COLOR"))
    )

def collapse_secondary_colors(col: pl.Expr) -> pl.Expr:
    c = (
        col.str.to_uppercase()
          .str.replace_all(r"[^\w\s/-]", "")   # drop odd chars like backslashes
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
    )
    # Normalize some aliases first
    c = (
        c.str.replace_all(r"\bBLUE\b", "GRAY")
         .str.replace_all(r"\bYELLOW\b", "ORANGE")
         .str.replace_all(r"\bBRN\b", "BROWN")
         .str.replace_all(r"\bORG\b", "ORANGE")
    )

    # Rule-based buckets
    return (
        pl.when(c.str.contains(r"\bBLACK"))      .then(pl.lit("BLACK"))
         .when(c.str.contains(r"\bWHITE"))       .then(pl.lit("WHITE"))
         .when(c.str.contains(r"\bGRAY"))        .then(pl.lit("GRAY"))
         .when(c.is_null())                              .then(pl.lit(None))
         .otherwise(pl.lit("OTHER COLOR"))
    )

def collapse_breeds(col: pl.Expr) -> pl.Expr:
    b = (
        col.str.to_uppercase()
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
    )
    # Keep hair-length domestics separate; bucket the rest
    base = (
        pl.when(b.str.starts_with("DOMESTIC SH")).then(pl.lit("DOMESTIC SH"))
         .when(b.str.starts_with("DOMESTIC MH")).then(pl.lit("DOMESTIC MH"))
         .when(b.str.starts_with("DOMESTIC LH")).then(pl.lit("DOMESTIC LH"))
    )

    return (
        pl.when(base.str.starts_with("DOMESTIC")).then(base)
         .otherwise(pl.lit("OTHER BREED"))
    )


columns_to_drop = [
    "Crossing",
    "Jurisdiction",
    "OutcomeSubtype", # All Nulls

]
columns_to_keep = [
    "LastUpdate",
    "AnimalID",
    "AnimalName",
    "AnimalType",
    "PrimaryColor",
    "SecondaryColor",
    "PrimaryBreed",
    "Sex",
    "DOB",
    "Age",
    "IntakeDate",
    "IntakeCondition",
    "IntakeType",
    "IntakeSubtype",
    "IntakeReason",
    "OutcomeDate",
    "OutcomeType",
    "OutcomeCondition",
]

# Check DOB vs Age -- Do they agree?
# EUTH Req? Do they always result in EUTH? / Same with DISPO REQ
# How many returns are there?
# IntakeReason into AnimalFault vs. Not Animal Fault
# Any Male/Neutered Pregnant?



# 


def print_missing_values(df: pl.DataFrame):
    counts = df.null_count()
    for col in counts.columns:
        print(f"{col} - {(counts[col][0] / len(df)) * 100:.1f}%")

def print_unique_values(df: pl.DataFrame):
    print("\n\nColumns + Values\n==============================")
    df = df.with_columns([
        pl.when(pl.col(col).is_in([True, False]))
        .then(pl.col(col).cast(pl.Utf8))
        .alias(col)
        for col in df.columns if df[col].dtype == pl.Boolean
    ])


    for column in df.columns:
        print(column, df[column].dtype)
        unique_values = df[column].unique(maintain_order=True).to_list()
        unique_value_counts = df[column].unique_counts().to_list()

        # sort values + unique_value_counts by counts descending (they go togehter)
        value_counts = zip(unique_values, unique_value_counts)
        # sort by counts
        sorted_value_counts = sorted(value_counts, key=lambda x: x[1], reverse=True)

        if len(unique_values) < 100:
            for value, count in sorted_value_counts:
                print(f"\t{value} - {count}")


def categorical_conditionals_text(
    df: pl.DataFrame,
    cat_cols: Iterable[str],
    include_nulls: bool = False,
    pct_decimals: int = 1,
) -> str:
    """
    For each categorical column, print how the other columns distribute
    (percentages) for each of its values, in a human-readable text format.
    Excludes 0% entries for clarity.
    """
    if not cat_cols:
        return "(no categorical columns provided)"

    lines = []

    def maybe_filter_nulls(d, cols):
        return d if include_nulls else d.drop_nulls(cols)

    for base_col in cat_cols:
        lines.append(f"\n=== {base_col} ===")
        base_values = (
            df.select(base_col).drop_nulls() if not include_nulls else df.select(base_col)
        )[base_col].unique().to_list()

        for base_val in base_values:
            subset = maybe_filter_nulls(df.filter(pl.col(base_col) == base_val), [base_col])
            total = len(subset)
            lines.append(f"\n{base_col} = {base_val!r} (n={total}):")

            if total == 0:
                lines.append("  (no rows)\n")
                continue

            for other_col in cat_cols:
                if other_col == base_col:
                    continue

                other_counts = (
                    maybe_filter_nulls(subset, [other_col])
                    .group_by(other_col)
                    .len()
                    .rename({"len": "count"})
                )

                other_counts = other_counts.with_columns(
                    ((pl.col("count") * 100 / total).round(pct_decimals)).alias("pct")
                )

                # Filter out 0% entries
                other_counts = other_counts.filter(pl.col("pct") > 0)

                if other_counts.height == 0:
                    continue  # skip entirely if no nonzero values

                lines.append(f"  {other_col}:")
                for row in other_counts.iter_rows(named=True):
                    lines.append(f"    - {row[other_col]!r}: {row['pct']:.{pct_decimals}f}%")

    return "\n".join(lines)

def percentiles_time_in_shelter(df: pl.DataFrame, percentiles: list[float]) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("OutcomeDate") - pl.col("IntakeDate")).dt.total_days().alias("TimeInShelterDays")
    )

    percentile_exprs = [
        pl.col("TimeInShelterDays").quantile(q).alias(f"P{int(q * 100)}") for q in percentiles
    ]

    result = df.select(percentile_exprs)

    return result


def preprocess():
    lfs: list[pl.LazyFrame] = []
    for file in glob.glob("data/raw/*.csv"):
        lf = pl.scan_csv(file, try_parse_dates=True)
        lfs.append(lf)

    lf = pl.concat(lfs)

    df = lf.collect()

    cols = set(df.columns)

    not_referenced_cols = cols.difference(columns_to_drop).difference(columns_to_keep)

    assert len(not_referenced_cols) == 0, (f"Columns not in REMOVE or KEEP! {list(not_referenced_cols)}")

    common_cols = set(columns_to_drop).intersection(columns_to_keep)
    assert len(common_cols) == 0, f"{list(common_cols)} are in BOTH REMOVE and KEEP!"

    df = df[columns_to_keep]

    rows_to_remove = {
        "IntakeType": ["SPAY", "NEUTER"]
    }

    df =df.filter(
        (pl.col("IntakeType").is_in(["SPAY", "NEUTER", "DISASTER", "DISPO REQ", "WILDLIFE", "S/N CLINIC"]).not_())
        &
        (pl.col("OutcomeType").is_in(["SPAY", "NEUTER", "FOUND ANIM", "LOST EXP", "FOUND EXP", "MISSING", "RTO", "REQ EUTH"]).not_())
        &
        (pl.col("AnimalType").eq("CAT")) 
        # & (pl.col("IntakeDate") < pl.datetime(2025, 1, 1))
        & (pl.col("OutcomeDate") < datetime.now() + timedelta(days=1)) # Anything in the future is wrong
    )

    df  = df.filter(
        pl.col("OutcomeType").is_in(["DIED"]).not_()
        )

    # df = df.filter(pl.col("IntakeCondition").eq("FERAL"))

    for state in ["Outcome", "Intake"]:
        df = df.with_columns(
            pl.when(pl.col(f"{state}Condition") == "NURSING")
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias(f"{state}IsNursing")
        )
        df = df.with_columns(
            pl.when(pl.col(f"{state}Condition") == "NORMAL").then(0)
            .when(pl.col(f"{state}Condition") == "BEH R").then(1)
            .when(pl.col(f"{state}Condition") == "BEH M").then(2)
            .when(pl.col(f"{state}Condition") == "BEH U").then(3)
            .otherwise(0)  # or .otherwise(999) for unknowns
            .alias(f"{state}BehaviorIssueIndex")
        )
        df = df.with_columns(
            pl.when(pl.col(f"{state}Condition") == "NORMAL").then(0)
            .when(pl.col(f"{state}Condition") == "MED R").then(1)
            .when(pl.col(f"{state}Condition") == "MED M").then(2)
            .when(pl.col(f"{state}Condition") == "MED SEV").then(3)
            .when(pl.col(f"{state}Condition") == "MED EMERG").then(4)
            .otherwise(statement=0)  # If it wasn't noted, then we assume NORMAL
            .alias(f"{state}MedicalIssueIndex")
        )

        df = df.drop(f"{state}Condition")

    
    df = df.with_columns(
        pl.max_horizontal(pl.col("IntakeDate"), pl.col("DOB")).alias("IntakeDate")
    )
    
    # Print number of Nones of DOB
    df = df.with_columns(
        (pl.col("IntakeDate") - pl.col("DOB")).dt.total_days().alias("AgeDays")
    )

    # df = df.drop(["DOB", "Age"])

    # --- 2) Apply to your dataframe ----------------------------------------------
# Suppose df has columns: "PrimaryColor", "SecondaryColor", "PrimaryBreed"

    df = df.with_columns([
        collapse_colors(pl.col("PrimaryColor")).alias("PrimaryColor"),
        collapse_secondary_colors(pl.col("SecondaryColor")).alias("SecondaryColor"),
        collapse_breeds(pl.col("PrimaryBreed")).alias("PrimaryBreed"),
    ])

    # Introduces target leakage! This is updated when they are adopted.
    # All adoptions are marked as spayed/neutered
    #
    # df = df.with_columns(
    #     pl.when(pl.col("Sex").is_in(["SPAYED", "NEUTERED"])).then(pl.lit("Spayed/Neutered"))
    #     .when(pl.col("Sex").is_in(["MALE", "FEMALE"])).then(pl.lit("Not Spayed/Neutered"))
    #     .otherwise(pl.lit("Unknown"))
    #     .alias("SpayedNeutered")
    # )

    df = df.with_columns(
            pl.when(pl.col("Sex").is_in(["MALE", "NEUTERED"])).then(pl.lit("Male"))
            .when(pl.col("Sex").is_in(["FEMALE", "SPAYED"])).then(pl.lit("Female"))
            .otherwise(pl.lit("Unknown"))
            .alias("Sex")
        )

    df = df.with_columns(
           pl.when(pl.col("AgeDays") < 365).then(pl.col("AgeDays")).alias("YoungCatAgeDays")
        )

    df = df.with_columns(
        pl.when(pl.col("AgeDays") >= 365).then((pl.col("AgeDays") / 365).round()).alias("OldCatAgeYears")
    )
    
    df = df.with_columns(
        pl.col("IntakeDate").dt.month().alias("IntakeMonth"),
    )
    
    df = df.with_columns(
        (pl.col("OutcomeDate") - pl.col("IntakeDate")).dt.total_days().alias("TimeInShelterDays")
    )
    pantab.frame_to_hyper(df, "data/clean/cleaned_data.hyper", table="records")

    # print_missing_values(df)
    print_unique_values(df)
    # print(categorical_conditionals_text(df, ["IntakeCondition", "OutcomeType", "OutcomeCondition"]))
    # with open("output.txt", "w") as f:
    #     _ = f.write(categorical_conditionals_text(df, ["IntakeType", "OutcomeType"]))

    df.write_parquet("data/clean/cleaned_data.parquet")


if __name__ == "__main__":

    preprocess()