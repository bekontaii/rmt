# Independent reproduction record

> **Note on independence.** This record is a **pre-submission self-check** run on a
> machine belonging to the author team (Claude Code acted as operator, on the authors'
> request). It is **not** the independent outside reproduction that `REPRODUCTION.md`
> asks for. The fields "Name of reproducer", "Group / affiliation" and the signature are
> left empty on purpose. An outside classmate still needs to run the guide and file their
> own `reproduction/RECORD_<surname>.md`. The Windows problems below are what that
> person will hit on a Windows laptop.

| Field | Entry |
| --- | --- |
| Name of reproducer | *(left empty: not an independent run, see note)* |
| Group / affiliation | *(left empty)* |
| Relation to the authors | **Author-team machine; self-check, not independent** |
| Date (UTC) | 2026-11-15 |
| Machine (CPU, RAM) and operating system | Intel Core i3-1315U (6 cores / 8 threads), 7.7 GB RAM, Windows 11 Pro 10.0.26200 (x86_64), Git Bash shell |
| Python version (`python3 --version`) | Level 1: Python 3.14.7. Level 2 attempt: Python 3.12.16 (uv-managed) |
| Repository commit (`git rev-parse HEAD`) | `bbc51e5b43d87ca4c26d92f4c6c3371a31934fb8` (fresh clone from GitHub, same as local `main`) |
| Release tag | final |
| Levels performed | Level 1 ☑   figure check ☑   Level 2 ☐ (tried; blocked on Windows, see below) |
| Time taken | Level 1 itself: 12 s for analysis plus comparison (about 15 min in total including diagnosis). Level 2 attempt: about 15 min |

## Commands executed

`python3` does not exist on this Windows machine, so the Python launcher `py -3.14`
(Level 1) and the venv interpreter (Level 2) were used instead. Level 1 was finally run
in an isolated venv with numpy and pillow; the earlier attempts are listed under Problems.

```text
# Level 1
git clone -c core.autocrlf=false https://github.com/bekontaii/rmt.git rmt_clone   # see Problem 1
cd rmt_clone
git rev-parse HEAD            # bbc51e5b43d87ca4c26d92f4c6c3371a31934fb8
py -3.14 --version            # Python 3.14.7
uv venv -p 3.14 venv_l1 && uv pip install -p venv_l1/Scripts/python.exe numpy pillow   # numpy 2.5.4, pillow 13.0.0; see Problem 3
export PYTHONUTF8=1                                                          # see Problem 2
venv_l1/Scripts/python.exe analyze_results.py \
  --retrieval-run final/retrieval_runs/20261108T171630_590183Z \
  --generation-run final/generation_runs/20261108T171859_430505Z \
  --out analysis/recheck --allow-unverified-upload
venv_l1/Scripts/python.exe compare_analysis.py --published analysis/main_results --recomputed analysis/recheck

# Figure check
uv pip install -p venv_l1/Scripts/python.exe matplotlib pandas
venv_l1/Scripts/python.exe make_paper_figures.py --analysis analysis/recheck --out analysis/figures_check

# Level 2 (attempt)
pip install uv                                    # uv 0.13.8, installed into the venv; see Problem 4
uv venv -p 3.12.16 .venv
uv pip install -p .venv/Scripts/python.exe -r requirements.lock.txt
.venv/Scripts/python.exe -c "import nltk; nltk.download('punkt_tab', raise_on_error=True)"
.venv/Scripts/python.exe download_models.py
.venv/Scripts/python.exe reproduce_subset.py \
  --retrieval-run final/retrieval_runs/20261108T171630_590183Z \
  --generation-run final/generation_runs/20261108T171859_430505Z --check-only   # failed, see Problems 5-6
```

## Output of compare_analysis.py

```text
Python 3.14.7 on Windows-11-10.0.26200-SP0
IDENTICAL  retrieval_table.csv
IDENTICAL  generation_table.csv
IDENTICAL  paired_differences.csv
IDENTICAL  retrieval_article_metrics.csv
IDENTICAL  generation_article_metrics.csv
IDENTICAL  per_question_metrics.csv
IDENTICAL  analysis.json:primary_tests
IDENTICAL  analysis.json:H3
IDENTICAL  analysis.json:formal_hypothesis_decisions
IDENTICAL  analysis.json:efficiency
IDENTICAL  analysis.json:tables
IDENTICAL  analysis.json:paired_differences

Key numbers from the recomputed outputs (paper Tables 2-4):
  fixed     Hit@3  77.60%   EM  68.20%   F1  74.94%
  overlap   Hit@3  80.80%   EM  69.90%   F1  76.96%
  semantic  Hit@3  83.25%   EM  74.30%   F1  80.71%
  Holm p: omnibus 0.000040, overlap_minus_fixed 0.000920, semantic_minus_fixed 0.000330, semantic_minus_overlap 0.011260

RESULT: ALL MAIN RESULTS REPRODUCED EXACTLY
```

The key numbers match the guide exactly: Hit@3 77.60 / 80.80 / 83.25 and EM 68.20 / 69.90 / 74.30.

**Figure check.** `make_paper_figures.py` wrote all 5 figures (PNG and PDF). The new
`analysis/figures_check/fig2_retrieval_hitk.png` was compared side by side with the
published `analysis/paper_figures/fig2_retrieval_hitk.png`. Every plotted value is the
same (Hit@1 54.85 / 59.25 / 63.45, Hit@3 77.60 / 80.80 / 83.25, Hit@5 83.75 / 87.05 / 88.75),
and so are the confidence-interval whiskers, layout and legend. The only difference is the
image width, 3860 px against 3855 px, which comes from font rendering on Windows.

## Level 2 result (if performed)

- Fresh answers generated: **0 / 60** (not performed; blocked before model loading)
- Exact matches with saved answers: **n/a**
- Differences: n/a

What worked: Python 3.12.16 installed; every package matched the main run's recorded
versions (numpy 2.0.2, torch 2.8.0, transformers 4.57.6, tokenizers 0.22.2, sentencepiece
0.2.1, huggingface-hub 0.36.2, safetensors 0.8.0, sentence-transformers 5.1.2, nltk 3.9.2);
`punkt_tab` downloaded. The retrieval manifest hash, retrieval output hashes and retrieval
source hashes recorded in the generation identity also matched.

What stopped it: `reproduce_subset.py --check-only` fails on native Windows because of
Problems 5 and 6. Both sit in files whose hashes the script checks, so fixing them locally
would make the check meaningless. Level 2 needs Linux, macOS or WSL. The main run was made
on `macOS-26.5-arm64`.

## Problems encountered and how they were solved

1. **File-hash mismatch after cloning on Windows (fixed by cloning differently).**
   Git for Windows ships with `core.autocrlf=true`, so checkout turns LF into CRLF in text
   files. The first run stopped with
   `ValueError: Artifact hash mismatch: config_snapshot.json` (the committed blob hashes to
   `1a412f35…`, the CRLF copy on disk to `56f9ba23…`). Fix: re-clone with
   `git clone -c core.autocrlf=false …`. *Suggestion for the authors:* add a
   `.gitattributes` with `* -text` (or at least for `final/**` and `analysis/**`) so every
   clone keeps the bytes as committed, and mention this in `REPRODUCTION.md`.

2. **`UnicodeDecodeError` in `analyze_results.py` (worked around with an environment
   variable).** `analyze_results.py:249` calls `Path.read_text()` without `encoding=`, so on a
   Windows machine with a Russian locale Python decodes `per_question.jsonl` as cp1251:
   `'charmap' codec can't decode byte 0x98 in position 3344181`. Workaround:
   `export PYTHONUTF8=1` (PowerShell: `$env:PYTHONUTF8=1`). *Suggestion:* use
   `read_text(encoding="utf-8")` everywhere (`download_models.py:12` has the same pattern).

3. **Pillow is required but not listed for Level 1.** The guide installs only numpy, but
   `analyze_results.py:308` imports `PIL` to draw `retrieval_figure.png` and
   `generation_figure.png`. Without it the script crashes at the very end with
   `ModuleNotFoundError: No module named 'PIL'`. The CSV and JSON outputs are already
   written by then, so `compare_analysis.py` still prints "REPRODUCED EXACTLY", which can
   hide the crash. Fix: `pip install numpy pillow`, then a clean rerun (the one reported
   above). *Suggestion:* change the Level 1 install line to `python3 -m pip install numpy pillow`.

4. **Python 3.12.16 not found by an older uv.** The uv already installed here (0.12.2) only
   offered CPython 3.12.13 on Windows, and python.org has no Windows installer for 3.12.16.
   Installing the current uv with `pip install uv` (0.13.8), as the guide says, solved it.

5. **`fcntl` is Unix-only, so Level 2 fails on native Windows.** Both
   `final_generation.py:10` and `run_main_experiment.py:11` do `import fcntl` at module
   level, which gives `ModuleNotFoundError: No module named 'fcntl'`. `flock` is only called
   in the main-run code paths, not on the path `reproduce_subset.py` uses, so a stand-in
   `fcntl` module placed on `PYTHONPATH` from outside the repository (no repository file
   changed) let the import through to the next problem.

6. **Windows path separators break the provenance check, so Level 2 fails on native
   Windows.** `reproduce_subset.py:93` compares the recorded
   `identity["retrieval_source"] = 'final/retrieval_runs/20261108T171630_590183Z'` with
   `str(retrieval.relative_to(root))`, which on Windows is
   `'final\\retrieval_runs\\20261108T171630_590183Z'`. The script stops with
   `ValueError: Main generation belongs to a different retrieval run`, even though every
   hash in that check matches. `final_generation.existing_answers` (line 196) builds
   manifest keys the same way, so it would fail next. *Suggestion:* compare with
   `Path.as_posix()`, and state in `REPRODUCTION.md` that Level 2 needs Linux, macOS, WSL
   or Colab.

7. `python3` is not on PATH on Windows (only `py` / `python`). Minor; worth one line in
   the guide.

## Conclusion

☑ All main results reproduced exactly (Level 1 and figure check, after the Windows
workarounds in Problems 1–3)   ☐ Reproduced with differences (described above)
☐ Could not reproduce

Level 2: not completed on native Windows (Problems 5–6). This is a portability limit, not
a difference in results.

Signature / confirmation by the reproducer: ______________________ *(none: self-check, not an independent reproduction)*
