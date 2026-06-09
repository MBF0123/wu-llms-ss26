# BSc Thesis — LLM-as-a-judge in Austrian Tax Law

## Pipeline 
1. code/build_annotation_sample.py — scores the model-answer pool with three
   LLM judges and selects 620 questions for annotation.
2. code/build_student_splits.py — divides the 620 questions into one file per
   student (circular-shift overlap: 60 questions each, 3 annotators per
   question). Corrected version: the earlier split padded short files with a
   repeated question; this one assigns 60 unique questions to every student.
3. code/annotation_quality_pipeline.py — audits the returned annotations and
   verifies each cited excerpt against the actual statute text.

## Data
- dataset_clean.csv, cleaned_dataset.csv — prompts and model answers
- Austrian_Tax_Law_Dataset.xlsx — reference answers and legal sources
- sample_620_v2.csv, annotator_assignments_blank.csv — sampling outputs
- annotation_task:student_files/ — the returned student annotations
