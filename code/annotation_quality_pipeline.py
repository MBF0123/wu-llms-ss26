"""
annotation_quality_pipeline.py
================================

Purpose
-------
Assess the *annotation quality* and *grounding consistency* of student
submissions for the Austrian Tax Law LLM-as-a-judge thesis.

This pipeline is intentionally simple and rule-based. It does NOT try to
detect AI use. It only checks whether annotations follow the formatting
guidelines and whether the cited grounding refers to the same legal
paragraph that is listed in the `sources` column.

Inputs
------
A folder of CSV files, one per student, named `student_<id>.csv`.
Each CSV is expected to contain at least these columns:
    row_key, sources, grounding_citations

Outputs (written to OUTPUT_DIR)
-------------------------------
1. combined_grounding_annotations.csv   - all rows from all students
2. grounding_quality_by_row.csv         - per-row flags
3. student_grounding_summary.csv        - per-student aggregates + label
4. examples_needing_review.csv          - small sample for manual review

Usage
-----
    python annotation_quality_pipeline.py

Adjust INPUT_DIR and OUTPUT_DIR below to match your setup.
"""

import os
import re
import glob
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder containing the student CSV files.
INPUT_DIR = "./student_files"

# Folder where the four output CSVs will be written.
OUTPUT_DIR = "./quality_outputs"

# Columns we actually need from each student CSV.
KEEP_COLUMNS = ["row_key", "sources", "grounding_citations"]

# Minimum number of annotated rows below which we do not trust the
# reliability label (we mark the student as 'likely unverified' instead).
MIN_ANNOTATED_ROWS = 5

# Thresholds for the cautious reliability label.
# A row is "passing" if it has no formatting issue AND its citation is
# consistent with the source label.
RELIABLE_THRESHOLD = 0.80    # >= 80% passing rows -> likely reliable
REVIEW_THRESHOLD   = 0.50    # >= 50% passing rows -> needs review
                             # < 50% passing rows -> likely unverified


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_student_id(filename: str) -> str:
    """Pull the student id out of a filename like 'student_12345678.csv'."""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]            # 'student_12345678'
    parts = name.split("_", 1)                  # ['student', '12345678']
    return parts[1] if len(parts) > 1 else name


def load_one_student(path: str) -> pd.DataFrame:
    """Read one student CSV, tag it with student_id, and return a slim frame."""
    df = pd.read_csv(path)

    # Make sure the columns we rely on exist; if not, create them as empty
    # so the rest of the pipeline does not crash on malformed submissions.
    for col in KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[KEEP_COLUMNS].copy()
    df.insert(0, "student_id", extract_student_id(path))
    return df


def is_blank(value) -> bool:
    """True if the cell is NaN, None, or contains only whitespace."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


# Quotation marks we accept. German legal writing uses „ … "  (U+201E … U+201C),
# but students sometimes use straight ASCII " or French « ». We treat
# *any* quote-like character as a delimiter and just require that text
# sits between two of them.
ANY_QUOTE_CHAR = '"\u201E\u201C\u201D\u00AB\u00BB\u2018\u2019' + "'"


def extract_quoted_excerpts(text: str) -> list[str]:
    """
    Pull out every substring that sits between a pair of quotation marks.
    Handles ASCII (" ") and German typographic quotes („ ").

    Strategy: any quote-like character, then a non-greedy run of
    non-quote characters, then any quote-like character.
    """
    if is_blank(text):
        return []

    qcls = "[" + re.escape(ANY_QUOTE_CHAR) + "]"
    not_q = "[^" + re.escape(ANY_QUOTE_CHAR) + "]+?"
    pattern = qcls + "(" + not_q + ")" + qcls
    matches = re.findall(pattern, str(text), flags=re.DOTALL)
    # Drop trivially short excerpts (<10 chars looks like an accidental quote).
    return [m.strip() for m in matches if len(m.strip()) >= 10]


def has_any_quotes(text: str) -> bool:
    """True if the cell contains at least one quotation mark character."""
    if is_blank(text):
        return False
    return any(ch in str(text) for ch in ANY_QUOTE_CHAR)


# A "legal paragraph reference" looks like §  9  Abs. 6  Z 4  KStG 1988
# We collect the *numeric* anchors (paragraph number, Abs, Z, lit) so we
# can compare references in `sources` against references in `grounding_citations`
# without being fooled by spacing or year suffixes.
PARAGRAPH_REGEX = re.compile(
    r"§\s*(\d+[a-zA-Z]?)"                     # § 9 / § 6a
    r"(?:\s*Abs\.?\s*(\d+))?"                # Abs. 6
    r"(?:\s*Z\s*(\d+))?"                     # Z 4
    r"(?:\s*lit\.?\s*([a-z]))?",             # lit. a
    flags=re.IGNORECASE,
)


def extract_paragraph_refs(text: str) -> set[str]:
    """
    Return the set of normalized paragraph references found in `text`.
    A reference is normalized to a string like '9|6|4|a' so two equal
    references compare equal regardless of spacing.
    """
    if is_blank(text):
        return set()

    refs = set()
    for m in PARAGRAPH_REGEX.finditer(str(text)):
        para, abs_, z, lit = m.groups()
        # Normalize each component to lowercase or empty string.
        key = "|".join([
            (para or "").lower(),
            (abs_ or ""),
            (z or ""),
            (lit or "").lower(),
        ])
        refs.add(key)
    return refs


def check_formatting(citations: str) -> dict:
    """
    Run the three formatting checks on one `grounding_citations` cell.
    Returns a dict of boolean flags.
    """
    flags = {
        "missing_citation": False,
        "no_quotation_marks": False,
        "malformed": False,
    }

    if is_blank(citations):
        flags["missing_citation"] = True
        return flags

    # Must contain at least one quotation character.
    if not has_any_quotes(citations):
        flags["no_quotation_marks"] = True

    # Must contain at least one well-formed quoted excerpt of reasonable length.
    excerpts = extract_quoted_excerpts(citations)
    if not excerpts:
        flags["malformed"] = True

    return flags


def check_grounding_consistency(sources: str, citations: str) -> dict:
    """
    Compare the legal paragraph references in `sources` against those in
    `grounding_citations`. Returns counts and a single boolean flag.

    Rule: at least one paragraph reference in `sources` must also appear in
    `grounding_citations`. If `sources` lists no parseable reference we
    cannot judge consistency and mark it as 'unverifiable'.
    """
    src_refs = extract_paragraph_refs(sources)
    cit_refs = extract_paragraph_refs(citations)

    if not src_refs:
        # No structured reference in `sources` -> cannot check automatically.
        return {
            "source_refs_count": 0,
            "citation_refs_count": len(cit_refs),
            "refs_overlap_count": 0,
            "citation_not_in_source": False,   # cannot judge
            "unverifiable_source": True,
        }

    overlap = src_refs & cit_refs
    return {
        "source_refs_count": len(src_refs),
        "citation_refs_count": len(cit_refs),
        "refs_overlap_count": len(overlap),
        "citation_not_in_source": len(overlap) == 0,
        "unverifiable_source": False,
    }


def row_needs_review(row: pd.Series) -> bool:
    """A row needs review if it has any formatting flag or a citation mismatch."""
    return bool(
        row["missing_citation"]
        or row["no_quotation_marks"]
        or row["malformed"]
        or row["citation_not_in_source"]
    )


def reliability_label(total: int, passing: int) -> str:
    """Cautious, rule-based reliability label per student."""
    if total < MIN_ANNOTATED_ROWS:
        return "likely unverified"
    share = passing / total
    if share >= RELIABLE_THRESHOLD:
        return "likely reliable"
    if share >= REVIEW_THRESHOLD:
        return "needs review"
    return "likely unverified"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1: discover files --------------------------------------------
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "student_*.csv")))
    if not files:
        raise FileNotFoundError(f"No student_*.csv files in {INPUT_DIR}")
    print(f"[1/8] Found {len(files)} student files.")

    # ---- Step 2: load & combine --------------------------------------------
    frames = [load_one_student(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    print(f"[2/8] Combined into one frame with {len(combined)} rows.")

    # ---- Step 3: drop completely empty rows --------------------------------
    # A row counts as "annotated" if at least `grounding_citations` is not blank.
    combined["is_annotated"] = combined["grounding_citations"].apply(
        lambda v: not is_blank(v)
    )
    print(
        f"[3/8] Of those, {combined['is_annotated'].sum()} rows actually "
        f"contain a grounding citation."
    )

    # Save the combined file (all rows, annotated or not) for auditing.
    combined_path = os.path.join(OUTPUT_DIR, "combined_grounding_annotations.csv")
    combined.to_csv(combined_path, index=False)

    # ---- Step 4 & 5: formatting + extracted excerpts -----------------------
    fmt = combined["grounding_citations"].apply(check_formatting).apply(pd.Series)
    excerpts = combined["grounding_citations"].apply(extract_quoted_excerpts)
    combined = pd.concat([combined, fmt], axis=1)
    combined["extracted_excerpts"] = excerpts
    combined["n_excerpts"] = excerpts.apply(len)
    print("[4/8] Formatting checks done.")
    print("[5/8] Quoted excerpts extracted.")

    # ---- Step 6: grounding consistency -------------------------------------
    consistency = combined.apply(
        lambda r: check_grounding_consistency(r["sources"], r["grounding_citations"]),
        axis=1,
    ).apply(pd.Series)
    combined = pd.concat([combined, consistency], axis=1)
    print("[6/8] Grounding-consistency checks done.")

    # ---- Step 7: flag suspicious rows + per-student summary ----------------
    combined["needs_review"] = combined.apply(row_needs_review, axis=1)

    # Row-level output (one row per annotation).
    row_cols = [
        "student_id", "row_key", "sources", "grounding_citations",
        "n_excerpts",
        "missing_citation", "no_quotation_marks", "malformed",
        "source_refs_count", "citation_refs_count", "refs_overlap_count",
        "citation_not_in_source", "unverifiable_source",
        "needs_review",
    ]
    row_quality = combined[combined["is_annotated"]][row_cols].copy()
    row_quality.to_csv(
        os.path.join(OUTPUT_DIR, "grounding_quality_by_row.csv"), index=False
    )

    # Student-level summary.
    grouped = row_quality.groupby("student_id")
    summary = pd.DataFrame({
        "total_rows":              grouped.size(),
        "missing_citations":       grouped["missing_citation"].sum(),
        "no_quotation_marks":      grouped["no_quotation_marks"].sum(),
        "malformed":               grouped["malformed"].sum(),
        "citation_not_in_source":  grouped["citation_not_in_source"].sum(),
        "unverifiable_source":     grouped["unverifiable_source"].sum(),
        "rows_needing_review":     grouped["needs_review"].sum(),
    }).reset_index()

    summary["passing_rows"] = summary["total_rows"] - summary["rows_needing_review"]
    summary["share_passing"] = (
        summary["passing_rows"] / summary["total_rows"]
    ).round(3)
    summary["reliability_label"] = summary.apply(
        lambda r: reliability_label(r["total_rows"], r["passing_rows"]),
        axis=1,
    )

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "student_grounding_summary.csv"), index=False
    )
    print("[7/8] Per-student summary written.")

    # ---- Step 8: examples for manual review --------------------------------
    # Take up to 3 problematic rows per student so the supervisor can sanity-check.
    examples = (
        row_quality[row_quality["needs_review"]]
        .groupby("student_id")
        .head(3)
        .copy()
    )
    examples.to_csv(
        os.path.join(OUTPUT_DIR, "examples_needing_review.csv"), index=False
    )
    print(f"[8/8] {len(examples)} example rows exported for manual review.")

    # ---- Final printout ----------------------------------------------------
    print("\n=== Reliability label distribution ===")
    print(summary["reliability_label"].value_counts().to_string())
    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
