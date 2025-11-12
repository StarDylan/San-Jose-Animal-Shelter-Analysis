from enum import unique
import glob
from collections.abc import Iterable
import polars as pl
import pantab
from datetime import datetime, timedelta

columns_to_drop = [
    "Crossing",
    "Jurisdiction",
    "OutcomeSubtype", # All Nulls
    "LastUpdate",

]
columns_to_keep = [
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


def preprocess():
    lfs: list[pl.LazyFrame] = []
    for file in glob.glob("../data/raw/*.csv"):
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

    # df  = df.filter(pl.col("OutcomeType").is_in(["TRANSFER", "DIED", "RESCUE", "RTF"]).not_())

    # pantab.frame_to_hyper(df, "../data/clean/all_data.hyper", table="records")

    # print_missing_values(df)
    print_unique_values(df)
    # print(categorical_conditionals_text(df, ["OutcomeCondition", "OutcomeType"], True))
    # with open("output.txt", "w") as f:
    #     _ = f.write(categorical_conditionals_text(df, ["IntakeType", "OutcomeType"]))


if __name__ == "__main__":
    preprocess()