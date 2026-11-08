# Document chunking in a local RAG pipeline

Bekarys Uzbekbayev and Yerassyl Kuanyshov, SE-2415 — Research Methods and Tools, Assignment 4.

Repository: https://github.com/bekontaii/rmt

The team reports that Yerassyl Kuanyshov tested the project successfully
on 8 November 2026 under Linux. Machine specifications, package versions,
exact commands and the numerical comparison log were not supplied;
Appendix C records the confirmation at that level of detail.
The public repository has been created; code upload and the assignment-4 tag
are being prepared.

The main experiment compares fixed chunks of 100 words, 100-word chunks with
20-word overlap, and semantic chunks with a 100-word cap. The same MPNet
encoder and FLAN-T5-Large generator are used for all three conditions.

The fixed sample contains 20 SQuAD 1.1 articles, 2,000 retrieval questions and
1,000 generation questions. Each question is paired across conditions:
6,000 retrieval records and 3,000 generated answers. Development pilot data
are separate and must not be combined with these main results.

## Protocol evidence and execution

`protocol/sealed_plan.json` records the prospective local seal. The original
data, model revisions, numerical source files, prompts, scoring rules and
analysis plan are checked against its hashes before execution and at
completion. They remain unchanged.

Moodle submission is **unverified**. The user reported 8 November 2026 at
10:00 Asia/Almaty, earlier than the local seal at 10:56. No Moodle receipt or
comparison of uploaded bytes resolves that chronology. The user subsequently
explicitly requested execution without further waiting. The additional
`run_main_experiment.py` launcher records this deviation before inference.
The record at `protocol/freeze_manifest.json` has status
`execution_requested_upload_unverified`; it is not an upload receipt and does
not establish compliance with the assignment's prior-upload requirement.
The original receipt-dependent runners and `freeze_gate.py` are preserved.

## Environment and data

Inference uses Python 3.12.16, CPU float32 and the versions pinned in
`requirements.lock.txt`. The observed main machine is an Apple M5 Mac with
16 GiB RAM. To create the environment on a compatible machine:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -c "import nltk; nltk.download('punkt_tab', raise_on_error=True)"
python download_models.py
```

The integrity checks require the recorded Python patch version, package
versions and English Punkt resource bytes. Platform differences must be
recorded rather than silently accepted as the original environment.

The source is the public SQuAD 1.1 development file. Its SHA256 is
`95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.
The provided `data/selection_manifest.json` fixes the exact article and
question IDs. Use the saved corpus and questions; do not redraw the sample.
`download_data.py` provides the original data access step for a clean copy.
Model names, pinned revisions and all settings are in `experiment_config.json`.

## Validate without inference

```bash
python run_main_experiment.py --check-only
python final_retrieval.py --check-only
python final_generation.py --check-config
```

These commands validate metadata and local provenance; they do not generate
answers. Cached models are required for the actual offline main run.

## Main execution and recovery

The actual main experiment is orchestrated by:

```bash
python -u run_main_experiment.py --execute-with-unverified-upload
```

This explicitly disclosed route does not claim a verified Moodle upload.
The pipeline first saves complete retrieval outputs and then generates the
1,000 answers for each method. `final/pipeline_manifest.json` and the
timestamped `final/main_experiment_*.log` show progress. Every generated
answer has its own checkpoint, prompt, token IDs, reference scores, timing
and checksum. Retain all failed or interrupted attempts.

The main retrieval run is:
`final/retrieval_runs/20261108T171630_590183Z`.
The main generation run is:
`final/generation_runs/20261108T171859_430505Z`.

To resume that generation run, or verify its completed checkpoints without
regenerating already saved answers:

```bash
python -u run_main_experiment.py --execute-with-unverified-upload \
  --resume-generation final/generation_runs/20261108T171859_430505Z
```

Use the resume command after interruption. A fresh launch without the resume
argument can create a separate generation run; it must not be treated as the
same attempt or used to select favourable outputs.

## Analysis and report

The primary metric is answer-span Hit@3 over 2,000 questions. Exact Match,
token F1 and visible evidence coverage use the fixed 1,000-question subset.
The frozen `analysis_plan.json` specifies article-cluster bootstrap intervals,
100,000-draw permutation/sign-flip tests and Holm correction. H3 is an
exploratory association, not a causal claim.

The completed analysis package includes `analyze_results.py`. After both
main manifests show `completed`, run:

```bash
python analyze_results.py \
  --retrieval-run final/retrieval_runs/20261108T171630_590183Z \
  --generation-run final/generation_runs/20261108T171859_430505Z \
  --out analysis/main_results \
  --allow-unverified-upload
```

The output folder must be new. The flag discloses missing Moodle evidence;
it does not relax completeness, identity, source-hash or metric checks.
Tables, raw per-question metrics, intervals, tests, figures and the
prespecified 30-question error-review sample are derived from saved results.
Error coding must be reviewed separately and must never change reference
answers or global scores.

The reported team test is attributed to Yerassyl Kuanyshov. Detailed test
metadata and numerical equality of the main table are not yet documented.
Record the real commit and `assignment-4` tag after uploading the code.
Internal numerical verification remains separate from the teammate report.

The prepared subset helper can first be checked without loading models:

```bash
python reproduce_subset.py \
  --retrieval-run final/retrieval_runs/20261108T171630_590183Z \
  --generation-run final/generation_runs/20261108T171859_430505Z \
  --check-only
```

For a real fresh subset run, use the same paths with `--execute` and supply
the actual operator's name through `--operator`. The helper selects the first
generation question in each article, generates 60 separate answers, and keeps
them under `reproduction/`. It reports matches and differences without
overwriting the main run or automatically claiming that the operator is an
independent classmate. Record the actual person, environment, commands and
outcome in Appendix C after the run.

## Rights

The code in this study is distributed under `LICENSE`. SQuAD, article texts,
model weights and third-party libraries retain their own rights and terms:

- Dataset: https://rajpurkar.github.io/SQuAD-explorer/
- Encoder: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
- Generator: https://huggingface.co/google/flan-t5-large

The code licence does not relicense those third-party materials. Environment
folders, downloaded model weights and private credentials are excluded from
the reproducibility archive.

## Paper figure source

`plot_final.py` is the exact ReportLab plotting source for the figures embedded
in the Word report. The generated vector SVGs and PNGs are retained in
`analysis/figures/`, together with the coordinate and source-hash manifest.
The paper rendering used the bundled ReportLab, Node.js and Sharp paths
recorded in that helper; `requirements-figures.txt` pins its Python library.
On another machine those optional rendering paths must be adapted. The
main `analyze_results.py` command regenerates the numerical figure data and
equivalent charts using the pinned base environment, without requiring the
optional paper renderer. This rendering difference does not change any score,
interval or statistical test.

## Team submission documents

The current report is `Assignment4_Uzbekbayev_Kuanyshov.docx`. The updated
team protocol copy is `protocol/Protocol_Freeze_Uzbekbayev_Kuanyshov.docx`.
These author metadata changes were recorded after execution. The original
`protocol/Protocol_Freeze.docx`, local seal, execution manifests and numerical
results remain unchanged. `protocol/team_metadata_amendment.json` records
the version link and the team-reported testing statement.
