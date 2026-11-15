# Independent reproduction guide

This guide is for a person **outside the author team** who checks that the main results of
the manuscript can be reproduced from this repository. Level 1 is required (about 10
minutes, any computer with Python 3.11 or newer; tested with 3.13). Level 2 is optional (about 30 minutes, needs
internet for a 3 GB model download).

Record what you did in a copy of `reproduction/RECORD_TEMPLATE.md` named
`reproduction/RECORD_<your-surname>.md` and send it to the authors. Report differences
honestly — a documented difference is more useful than an unverified "it works".

## Level 1 — recompute every table and statistical test from the saved runs

No models are needed; this re-derives Tables 2–6, all p-values, intervals, and H3 from the
3,000 saved answers and 6,000 retrieval records, after re-checking every file hash.

```bash
git clone https://github.com/bekontaii/rmt.git
cd rmt
git rev-parse HEAD                      # copy this commit hash into your record
python3 --version                       # copy into your record
python3 -m pip install numpy
python3 analyze_results.py \
  --retrieval-run final/retrieval_runs/20261108T171630_590183Z \
  --generation-run final/generation_runs/20261108T171859_430505Z \
  --out analysis/recheck --allow-unverified-upload
python3 compare_analysis.py --published analysis/main_results --recomputed analysis/recheck
```

Copy the whole output of the last command into your record. The expected last line is
`RESULT: ALL MAIN RESULTS REPRODUCED EXACTLY`, and the key numbers must be
Hit@3 77.60 / 80.80 / 83.25 and EM 68.20 / 69.90 / 74.30.

Optional figure check:

```bash
python3 -m pip install matplotlib pandas
python3 make_paper_figures.py --analysis analysis/recheck --out analysis/figures_check
```

Open `analysis/figures_check/fig2_retrieval_hitk.png` and compare it with Figure 2 of the paper.

**On Google Colab:** run the same commands in cells, each prefixed with `!` (and `%cd rmt`
instead of `cd rmt`).

## Level 2 (optional) — generate 60 fresh answers and compare with the saved ones

This re-runs FLAN-T5-Large on one question per article for all three strategies, using the
saved retrieval results. It requires the exact Python version and packages of the main run
(the script refuses to run otherwise).

```bash
python3 -m pip install uv
uv venv -p 3.12.16 .venv && source .venv/bin/activate
uv pip install -r requirements.lock.txt
python -c "import nltk; nltk.download('punkt_tab', raise_on_error=True)"
python download_models.py
python reproduce_subset.py \
  --retrieval-run final/retrieval_runs/20261108T171630_590183Z \
  --generation-run final/generation_runs/20261108T171859_430505Z --check-only
python reproduce_subset.py \
  --retrieval-run final/retrieval_runs/20261108T171630_590183Z \
  --generation-run final/generation_runs/20261108T171859_430505Z \
  --operator "Your Full Name" --execute --out reproduction/$(date -u +%Y%m%d)_<surname>
```

The script reports how many of the 60 fresh answers match the saved ones exactly. Small
differences can occur on a different CPU architecture; report the count either way and
attach the output folder.
