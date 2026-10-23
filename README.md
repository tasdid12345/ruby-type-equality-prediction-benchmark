# Ruby parameter type-equality prediction

Question: can a class-balanced LightGBM model with richer name/context features predict whether two Ruby parameters (from ruby/rbs core and stdlib signatures) have equal declared types better than a name-similarity threshold baseline, on held-out signature files?

- Baseline: TF-IDF character n-gram cosine similarity of parameter names with a threshold tuned on training pairs.
- Improved: class-balanced LightGBM on name/context features, hyperparameters tuned with grouped cross-validation on the training set only.
- Split: by signature file (75/25), seed 42. CPU only.

Run: `pip install -r requirements.txt && python analysis.py`. Outputs `results.json` and `figures/comparison.png`.

## Results (latest run)

On held-out parameter pairs from ruby/rbs core and stdlib signatures, a standard name-similarity threshold predictor reached 67% F1 for the equal-types class, while my class-balanced LightGBM with richer name and context features reached 73%, a modest gain of about 6 points.

| Metric | Value |
|---|---|
| baseline_score | 0.6667 |
| improved_score | 0.7267 |
| improvement | 0.06001 |
| baseline_score_std | 0 |
| improved_score_std | 0.00116 |
| baseline_threshold | 0 |
| improved_threshold | 0.33 |
| cv_auc_train_best | 0.7655 |
| n_records | 4181 |
| baseline_f1 | 0.6667 |
| improved_f1 | 0.7267 |
| baseline_precision | 0.5 |
| improved_precision | 0.6284 |
| baseline_recall | 1 |
| improved_recall | 0.8615 |
| baseline_auc | 0.6473 |
| improved_auc | 0.8184 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Params parsed by regex from RBS (single-line defs, 'untyped' params dropped, types compared as exact strings). Pairs are sampled 50/50 within each file-grouped partition, so results depend on this synthetic balance. Train priors are out-of-fold by file; grouped CV reuses them, so CV is slightly optimistic. Baseline is deterministic (std 0); LightGBM averaged over seeds 42-44. Test set used once.

## Reference

This project is motivated by the following paper:

> M Kazerounian, JS Foster, B Min (2021). *SimTyper: sound type inference for Ruby using type equality prediction*. Proceedings of the ACM on. https://dl.acm.org/doi/abs/10.1145/3485483

Paper author: Prof. Jeff Foster, Tufts University (jfoster@cs.tufts.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
