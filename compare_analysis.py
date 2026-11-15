"""Compare a freshly recomputed analysis folder with the published one.

Usage (from the repository root, after running analyze_results.py into analysis/recheck):
    python compare_analysis.py --published analysis/main_results --recomputed analysis/recheck

Prints IDENTICAL/DIFFERENT for every table and statistical section and the key paper numbers,
so an independent reproducer can paste the output into reproduction/RECORD.md. Uses only the
Python standard library.
"""
import argparse, csv, hashlib, json, platform, sys
from pathlib import Path

TABLES = ["retrieval_table.csv", "generation_table.csv", "paired_differences.csv",
          "retrieval_article_metrics.csv", "generation_article_metrics.csv", "per_question_metrics.csv"]
SECTIONS = ["primary_tests", "H3", "formal_hypothesis_decisions", "efficiency", "tables", "paired_differences"]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--published", type=Path, default=Path("analysis/main_results"))
    ap.add_argument("--recomputed", type=Path, default=Path("analysis/recheck"))
    a = ap.parse_args(); ok = True
    print(f"Python {platform.python_version()} on {platform.platform()}")
    for f in TABLES:
        same = sha(a.published / f) == sha(a.recomputed / f); ok &= same
        print(f"{'IDENTICAL' if same else 'DIFFERENT'}  {f}")
    pa, ra = (json.loads((d / "analysis.json").read_text()) for d in (a.published, a.recomputed))
    for k in SECTIONS:
        same = pa.get(k) == ra.get(k); ok &= same
        print(f"{'IDENTICAL' if same else 'DIFFERENT'}  analysis.json:{k}")
    rows = {r["strategy"]: r for r in csv.DictReader(open(a.recomputed / "retrieval_table.csv"))}
    gen = {r["strategy"]: r for r in csv.DictReader(open(a.recomputed / "generation_table.csv"))}
    print("\nKey numbers from the recomputed outputs (paper Tables 2-4):")
    for s in ["fixed", "overlap", "semantic"]:
        print(f"  {s:9s} Hit@3 {float(rows[s]['hit_at_3'])*100:6.2f}%   EM {float(gen[s]['exact_match'])*100:6.2f}%   F1 {float(gen[s]['f1'])*100:6.2f}%")
    adj = ra["primary_tests"]["adjusted_p"]
    print("  Holm p: " + ", ".join(f"{k} {v:.6f}" for k, v in adj.items()))
    print("\nRESULT:", "ALL MAIN RESULTS REPRODUCED EXACTLY" if ok else "DIFFERENCES FOUND - report them in RECORD.md")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
