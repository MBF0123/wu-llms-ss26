"""
build_annotation_sample.py
==========================

Thesis pipeline: Austrian Tax Law annotation sample builder.

Takes model answers + prompts + a reference-answer/legal-source dataset, scores
a stratified candidate pool with three LLM judges (OpenAI, Gemini, Groq), then
selects exactly 620 diverse rows for human annotation and creates a blank
assignment file with three annotator slots per row.

Outputs (all written by this script):
    - sample_620_v2.csv
    - annotator_assignments_blank.csv
    - pipeline_explainer.txt

Run:
    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...
    export GROQ_API_KEY=...
    python build_annotation_sample.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Input files
    cleaned_dataset: str = "cleaned_dataset.csv"
    dataset_clean: str = "dataset_clean.csv"
    excel_dataset: str = "Austrian Tax Law Dataset.xlsx"  # NEW: reference answers + sources
    excel_sheet: str = "Dataset"                           # NEW: sheet to load

    # Output files
    out_sample: str = "sample_620_v2.csv"
    out_assignments: str = "annotator_assignments_blank.csv"
    out_explainer: str = "pipeline_explainer.txt"          # NEW: script now writes this itself
    out_checkpoint: str = "judge_scores_checkpoint.csv"

    # Sampling
    final_sample_size: int = 620
    candidate_pool_size: int = 1000
    annotator_slots: int = 3
    random_seed: int = 42

    # Judge / API behaviour
    api_timeout_s: int = 30
    api_max_retries: int = 3
    api_backoff_s: float = 2.0

    # Bucket mix (must sum to 1.0)
    share_disagreement: float = 0.35
    share_borderline: float = 0.25
    share_high_quality: float = 0.25
    share_random: float = 0.15


CFG = Config()
log = logging.getLogger("sample_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# Step 1 — Load & merge input datasets (CSVs + Excel)
# ---------------------------------------------------------------------------

def load_and_merge(cfg: Config) -> pd.DataFrame:
    """Load both CSVs and attach reference_answer + sources from the Excel file."""
    answers = pd.read_csv(cfg.cleaned_dataset)
    prompts = pd.read_csv(cfg.dataset_clean)

    # Normalise column names (dataset_clean has a BOM on the id column)
    prompts.columns = [c.strip().lstrip("\ufeff") for c in prompts.columns]

    merged = answers.merge(prompts, on="id", how="inner")

    # --- NEW: attach reference_answer and sources from the Excel dataset ---
    ref = pd.read_excel(cfg.excel_dataset, sheet_name=cfg.excel_sheet)
    ref.columns = [c.strip().lstrip("\ufeff") for c in ref.columns]
    # Excel column is `correct_answer` — rename for clarity downstream.
    ref = ref.rename(columns={"correct_answer": "reference_answer"})
    ref = ref[["id", "reference_answer", "sources"]].drop_duplicates(subset="id")

    # Left join: keep all answers even when reference data is missing.
    merged = merged.merge(ref, on="id", how="left")
    merged["reference_answer"] = merged["reference_answer"].fillna("")
    merged["sources"] = merged["sources"].fillna("")

    log.info("Merged dataset: %d rows, %d unique ids", len(merged), merged["id"].nunique())
    return merged


# ---------------------------------------------------------------------------
# Step 2 — Build a stable, unique row_key
# ---------------------------------------------------------------------------

def add_row_key(df: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic unique identifier for every row."""
    def _hash(row: pd.Series) -> str:
        raw = f"{row['id']}|{row['submission_folder']}|{row['model_file']}|{row.name}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    df = df.copy()
    df["row_key"] = df.apply(_hash, axis=1)
    df = df.drop_duplicates(subset="row_key").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 3 — LLM judges
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for Austrian tax law answers. "
    "Grade the ANSWER to the QUESTION on a 0-5 integer scale where "
    "0 = no answer, 1 = wrong/irrelevant, 5 = fully correct and well-argued. "
    "Respond with ONLY the single digit."
)


def _parse_score(raw: str) -> float:
    for ch in str(raw):
        if ch in "12345":
            return float(ch)
    return float("nan")  # unparseable → NaN, not neutral


def _retry(fn: Callable, *args, **kwargs):
    for attempt in range(1, CFG.api_max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if attempt == CFG.api_max_retries:
                log.warning("API call failed after %d attempts: %s", attempt, e)
                return None
            time.sleep(CFG.api_backoff_s * attempt)


def judge_openai(prompt: str, answer: str) -> Optional[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=CFG.api_timeout_s)

    def _call():
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"QUESTION:\n{prompt}\n\nANSWER:\n{answer}"},
            ],
            temperature=0,
            max_tokens=4,
        )
        return _parse_score(resp.choices[0].message.content)

    return _retry(_call)


def judge_gemini(prompt: str, answer: str) -> Optional[float]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=JUDGE_SYSTEM_PROMPT)

    def _call():
        resp = model.generate_content(
            f"QUESTION:\n{prompt}\n\nANSWER:\n{answer}",
            request_options={"timeout": CFG.api_timeout_s},
        )
        return _parse_score(resp.text)

    return _retry(_call)


def judge_groq(prompt: str, answer: str) -> Optional[float]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq
    client = Groq(api_key=api_key, timeout=CFG.api_timeout_s)

    def _call():
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"QUESTION:\n{prompt}\n\nANSWER:\n{answer}"},
            ],
            temperature=0,
            max_tokens=4,
        )
        return _parse_score(resp.choices[0].message.content)

    return _retry(_call)


JUDGES: dict[str, Callable[[str, str], Optional[float]]] = {
    "openai": judge_openai,
    "gemini": judge_gemini,
    "groq":   judge_groq,
}


# ---------------------------------------------------------------------------
# Step 4 — Candidate pool + judge scoring
# ---------------------------------------------------------------------------

def pick_candidate_pool(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Stratified random pool sampled per (id, model_file) so that high-volume
    prompts/models don't dominate the candidate set."""
    n = min(cfg.candidate_pool_size, len(df))
    frac = n / len(df)
    pool = (
        df.groupby(["id", "model_file"], group_keys=False)
          .apply(lambda g: g.sample(frac=frac, random_state=cfg.random_seed))
    )
    # Correct rounding drift: top-up if short, trim if over.
    if len(pool) < n:
        extra = df.drop(pool.index).sample(n=n - len(pool), random_state=cfg.random_seed)
        pool = pd.concat([pool, extra])
    elif len(pool) > n:
        pool = pool.sample(n=n, random_state=cfg.random_seed)
    return pool.reset_index(drop=True)


def _checkpoint_path(pool: pd.DataFrame, cfg: Config) -> Path:
    """Fingerprinted checkpoint path — changes to config/inputs/prompt force a
    fresh scoring run instead of silently reusing stale cached scores."""
    fingerprint = hashlib.sha1(
        "|".join([
            str(cfg.random_seed),
            str(cfg.candidate_pool_size),
            str(len(pool)),
            JUDGE_SYSTEM_PROMPT,
            ",".join(sorted(JUDGES.keys())),
        ]).encode("utf-8")
    ).hexdigest()[:10]
    p = Path(cfg.out_checkpoint)
    return p.with_name(f"{p.stem}_{fingerprint}{p.suffix}")


def score_with_judges(pool: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Ask each judge for a 1-5 score; store as columns score_<judge>."""
    ckpt = _checkpoint_path(pool, cfg)
    if ckpt.exists():
        log.info("Loading cached judge scores from %s", ckpt)
        return pd.read_csv(ckpt)

    scored = pool.copy()
    for name in JUDGES:
        scored[f"score_{name}"] = pd.NA

    for i in tqdm(range(len(scored)), desc="Scoring with judges"):
        prompt = str(scored.at[i, "prompt"])
        answer = str(scored.at[i, "answer"])
        for name, judge in JUDGES.items():
            scored.at[i, f"score_{name}"] = judge(prompt, answer)
        if (i + 1) % 100 == 0:
            scored.to_csv(ckpt, index=False)

    scored.to_csv(ckpt, index=False)
    return scored


def add_judge_stats(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean score and disagreement across judges. Missing scores stay NaN so
    API failures do NOT get imputed as neutral 3.0 and bias the borderline
    bucket. Rows with fewer than 2 valid scores are excluded from score-driven
    buckets (they can still be picked by the random bucket)."""
    score_cols = [c for c in scored.columns if c.startswith("score_")]
    scored = scored.copy()
    nums = scored[score_cols].apply(pd.to_numeric, errors="coerce")

    scored["valid_judge_count"] = nums.notna().sum(axis=1)
    scored["judge_mean"] = nums.mean(axis=1)             # skipna=True by default
    scored["judge_disagreement"] = nums.max(axis=1) - nums.min(axis=1)

    insufficient = scored["valid_judge_count"] < 2
    scored.loc[insufficient, ["judge_mean", "judge_disagreement"]] = pd.NA
    return scored


# ---------------------------------------------------------------------------
# Step 5 — Bucket-based sampling with deduplication and top-up
# ---------------------------------------------------------------------------

def sample_buckets(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = random.Random(cfg.random_seed)  # noqa: F841  (kept for future use)
    total = cfg.final_sample_size

    n_dis = int(total * cfg.share_disagreement)
    n_bor = int(total * cfg.share_borderline)
    n_hi  = int(total * cfg.share_high_quality)
    n_rnd = total - n_dis - n_bor - n_hi

    used: set[str] = set()
    picked: list[pd.DataFrame] = []

    def _take(ordered: pd.DataFrame, n: int, label: str) -> None:
        fresh = ordered[~ordered["row_key"].isin(used)].head(n)
        used.update(fresh["row_key"].tolist())
        picked.append(fresh.assign(bucket=label))

    # 1. Disagreement — judges differ the most (NaN rows fall to the end)
    _take(scored.sort_values("judge_disagreement", ascending=False, na_position="last"),
          n_dis, "disagreement")
    # 2. Borderline — mean score closest to 3 (NaN rows excluded)
    borderline = (scored.dropna(subset=["judge_mean"])
                        .assign(_dist=lambda d: (d["judge_mean"] - 3.0).abs())
                        .sort_values("_dist"))
    _take(borderline, n_bor, "borderline")
    # 3. High quality — highest mean score
    _take(scored.sort_values("judge_mean", ascending=False, na_position="last"),
          n_hi, "high_quality")
    # 4. Random — everything else
    pool_rest = scored[~scored["row_key"].isin(used)]
    _take(pool_rest.sample(frac=1, random_state=cfg.random_seed), n_rnd, "random")

    sample = pd.concat(picked, ignore_index=True).drop_duplicates(subset="row_key")

    # Top-up any shortfall from remaining scored rows
    if len(sample) < total:
        gap = total - len(sample)
        leftovers = scored[~scored["row_key"].isin(sample["row_key"])]
        if len(leftovers) < gap:
            raise RuntimeError(
                f"Cannot reach {total} unique rows; only {len(sample) + len(leftovers)} available."
            )
        filler = leftovers.sample(n=gap, random_state=cfg.random_seed).assign(bucket="topup")
        sample = pd.concat([sample, filler], ignore_index=True)

    assert len(sample) == total, f"expected {total}, got {len(sample)}"
    assert sample["row_key"].is_unique, "row_key duplicates in final sample"
    return sample.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 6 — Write the three deliverables
# ---------------------------------------------------------------------------

def write_sample_file(sample: pd.DataFrame, cfg: Config) -> None:
    """Output file 1 — 620 rows ready for review (now incl. reference data)."""
    cols = [
        "row_key", "id", "prompt", "answer",
        "submission_folder", "model_file",
        "reference_answer", "sources",        # NEW: legal reference material
    ]
    internal = [c for c in ("judge_mean", "judge_disagreement", "bucket") if c in sample.columns]
    sample[cols + internal].to_csv(cfg.out_sample, index=False)
    log.info("Wrote %s (%d rows)", cfg.out_sample, len(sample))


def write_assignment_file(sample: pd.DataFrame, cfg: Config) -> None:
    """Output file 2 — task_id × annotator_slot × blank student_id × row_key × prompt.
    Reference_answer and sources are intentionally NOT included here."""
    rows = []
    for task_id, row in enumerate(sample.itertuples(index=False), start=1):
        for slot in range(1, cfg.annotator_slots + 1):
            rows.append({
                "task_id":        task_id,
                "annotator_slot": slot,
                "student_id":     "",
                "row_key":        row.row_key,
                "prompt":         row.prompt,
            })
    assignments = pd.DataFrame(rows)
    expected = len(sample) * cfg.annotator_slots
    assert len(assignments) == expected, f"expected {expected}, got {len(assignments)}"
    assignments.to_csv(cfg.out_assignments, index=False)
    log.info("Wrote %s (%d rows)", cfg.out_assignments, len(assignments))


EXPLAINER_TEXT = """\
Pipeline: Austrian Tax Law annotation sample builder.

1. Loads cleaned_dataset.csv (model answers), dataset_clean.csv (prompts),
   and Austrian Tax Law Dataset.xlsx (reference answers + legal sources),
   and merges them on `id`. A stable row_key is built from id,
   submission_folder, model_file, and row index.
2. A stratified candidate pool (per id × model_file) is scored by three LLM
   judges (OpenAI, Gemini, Groq). Each judge returns an integer 1-5.
   Missing scores remain NaN — failed API calls are NOT imputed as neutral.
3. From the scored pool we select exactly 620 unique rows using four buckets:
   disagreement, borderline, high-quality, and random. Overlap is removed
   and any shortfall is topped up automatically.
4. Outputs:
     - sample_620_v2.csv              : 620 rows (incl. reference_answer, sources)
     - annotator_assignments_blank.csv: 620 x 3 = 1860 rows, student_id blank
     - pipeline_explainer.txt         : this summary
"""


def write_explainer(cfg: Config) -> None:
    """Output file 3 — short pipeline description, written by the script itself."""
    Path(cfg.out_explainer).write_text(EXPLAINER_TEXT, encoding="utf-8")
    log.info("Wrote %s", cfg.out_explainer)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(CFG.random_seed)

    merged = load_and_merge(CFG)
    merged = add_row_key(merged)

    pool = pick_candidate_pool(merged, CFG)
    scored = score_with_judges(pool, CFG)
    scored = add_judge_stats(scored)

    sample = sample_buckets(scored, CFG)

    write_sample_file(sample, CFG)
    write_assignment_file(sample, CFG)
    write_explainer(CFG)

    log.info("Done.  Final sample = %d unique rows.", len(sample))


if __name__ == "__main__":
    main()
