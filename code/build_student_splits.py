"""
================================================================================
Build per-student annotation files (corrected circular-shift split)
================================================================================

PURPOSE
-------
Takes the sampled questions and divides them into one file per student, so that:
  - every student receives exactly WINDOW unique questions (no duplicates,
    no padding), and
  - each question is annotated by ANNOTATIONS_PER_ROW different students,
    using the overlapping "circular shift" assignment.

This is the corrected version of the original split step. The original step
had a bug: when a student was assigned fewer than WINDOW questions, it padded
the file up to WINDOW rows by repeating a single question. This version never
pads — it assigns WINDOW unique rows to every student by wrapping around the
end of the sample (modulo arithmetic), exactly as the assignment design intended.

ASSIGNMENT DESIGN (from the supervisor's specification)
-------------------------------------------------------
  - WINDOW = 60 questions per student
  - ANNOTATIONS_PER_ROW = 3 students annotate each question
  - OFFSET = WINDOW / ANNOTATIONS_PER_ROW = 20  (each student starts 20 rows
    after the previous one)

  Student 1 -> rows 1..60
  Student 2 -> rows 21..80
  Student 3 -> rows 41..100
  ...
  The last students wrap around to the start of the sample, so rows near the
  beginning are also covered three times.
"""

import os
import pandas as pd

# ------------------------------------------------------------------ config ----
SAMPLE_PATH = "sample_620_v2.csv"     # the sampled questions
OUTPUT_DIR  = "student_splits"        # one CSV per student is written here

WINDOW              = 60              # questions assigned to each student
ANNOTATIONS_PER_ROW = 3               # how many students annotate each question
OFFSET              = WINDOW // ANNOTATIONS_PER_ROW   # = 20

# The list of student IDs (one file is produced per ID, in this order).
# Replace with the official list shared by the supervisor when you run it.
STUDENT_IDS = [
    "11805796", "11817953", "12044122", "12117106", "12127047", "12132051",
    "12211175", "12215660", "12220432", "12222912", "12227337", "12231895",
    "12239827", "12309335", "12311325", "12314670", "12317812", "12319305",
    "12328633", "12329053", "12331507", "12337808", "12344133", "12406664",
    "12407101", "12413051", "12416171", "12420172", "12420601", "12433533",
    "52006411",
]

# Columns the student receives (carried over from the sample) ...
CARRY_COLUMNS = ["row_key", "prompt", "answer", "reference_answer", "sources"]
# ... and the empty columns the student fills in.
BLANK_COLUMNS = [
    "correctness_score", "groundedness_score", "legal_reasoning_score",
    "completeness_score", "comprehension_score", "hallucination_flag",
    "grounding_citations", "comments",
]


def build_splits(sample_path=SAMPLE_PATH, student_ids=STUDENT_IDS,
                 window=WINDOW, offset=OFFSET, output_dir=OUTPUT_DIR):
    sample = pd.read_csv(sample_path)
    n = len(sample)
    os.makedirs(output_dir, exist_ok=True)

    for student_index, student_id in enumerate(student_ids):
        start = student_index * offset
        # Circular-shift indices: wrap around with modulo so every student
        # gets `window` UNIQUE rows even at the end of the list. This is the
        # line that replaces the old "pad to 60 with a repeated question" bug.
        indices = [(start + j) % n for j in range(window)]

        rows = sample.iloc[indices].reset_index(drop=True)

        out = pd.DataFrame()
        out["task_id"] = range(1, window + 1)          # 1..60 within the file
        for col in CARRY_COLUMNS:
            out[col] = rows[col].values if col in rows else ""
        for col in BLANK_COLUMNS:
            out[col] = ""                               # student fills these in

        out.to_csv(os.path.join(output_dir, f"student_{student_id}.csv"),
                   index=False)

    return sample, n


def verify(sample_path=SAMPLE_PATH, student_ids=STUDENT_IDS,
           window=WINDOW, offset=OFFSET):
    """Confirm: (1) every student gets `window` UNIQUE rows, (2) coverage."""
    n = len(pd.read_csv(sample_path))
    coverage = {}                                       # row_index -> #students
    all_unique = True
    for student_index in range(len(student_ids)):
        start = student_index * offset
        idx = [(start + j) % n for j in range(window)]
        if len(set(idx)) != window:                     # within-student dupes?
            all_unique = False
        for i in idx:
            coverage[i] = coverage.get(i, 0) + 1

    print(f"sample size: {n} rows | students: {len(student_ids)}")
    print(f"every student gets {window} UNIQUE rows (no padding): {all_unique}")
    dist = {}
    for c in coverage.values():
        dist[c] = dist.get(c, 0) + 1
    print(f"rows covered, by number of annotators: {dict(sorted(dist.items()))}")
    print(f"rows never covered: {n - len(coverage)}")


if __name__ == "__main__":
    build_splits()
    verify()
    print("Done. One file per student written to ./student_splits/")
