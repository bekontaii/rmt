# Level 1 reproduction record for Mustafa

**Status: Level 1 completed.** The supplied PowerShell screenshots show a completed recomputation and 12 identical comparisons.

| Field | Entry |
| --- | --- |
| Name of reproducer | Mustafa Nuradilet (Мустафа Нурадилет) |
| Group / affiliation | |
| Relation to the authors | Classmate outside the author team. Not a co-author. |
| Date (UTC) | 2026-11-15 09:00 UTC (14:00 Asia/Almaty) |
| Machine (CPU, RAM) and operating system | Windows 11 (10.0.26300), PowerShell. CPU and RAM not recorded in the supplied terminal output. |
| Python version (`python3 --version`) | Python 3.14.4 (virtual environment), NumPy 2.5.4, Pillow 13.0.0 |
| Repository commit (`git rev-parse HEAD`) | `bbc51e5b43d87ca4c26d92f4c6c3371a31934fb8` |
| Release tag at the time of the run, if any | |
| Levels performed | Level 1 ☑   figure check ☐   Level 2 ☐ |
| Time taken | |

## Commands executed

```text
git clone -c core.autocrlf=false https://github.com/bekontaii/rmt.git rmt-level1
cd rmt-level1
git rev-parse HEAD
py -3 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install numpy pillow
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe analyze_results.py --retrieval-run final/retrieval_runs/20261108T171630_590183Z --generation-run final/generation_runs/20261108T171859_430505Z --out analysis/recheck --allow-unverified-upload
.\.venv\Scripts\python.exe compare_analysis.py --published analysis/main_results --recomputed analysis/recheck
```

The recomputation reported mode `final`, 2,000 retrieval questions, 1,000 generation questions, output directory `analysis/recheck`, and `formal_decisions: true`.

## Output of compare_analysis.py

```text
Python 3.14.4 on Windows-11-10.0.26300-SP0
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

## Level 2 result (if performed)

No Level 2 run is recorded in the supplied evidence.

## Problems encountered and how they were solved

The supplied transcript shows a fresh clone with LF line endings, UTF-8 mode enabled in PowerShell, and the installation of NumPy and Pillow. The analysis and comparator both produced results.

## Conclusion

☑ All main results reproduced exactly   ☐ Reproduced with differences (described above)   ☐ Could not reproduce

Signature / confirmation by the reproducer: ______________________
