# Ruby parameter type-equality prediction

Question: can a class-balanced LightGBM model with richer name/context features predict whether two Ruby parameters (from ruby/rbs core and stdlib signatures) have equal declared types better than a name-similarity threshold baseline, on held-out signature files?

- Baseline: TF-IDF character n-gram cosine similarity of parameter names with a threshold tuned on training pairs.
- Improved: class-balanced LightGBM on name/context features, hyperparameters tuned with grouped cross-validation on the training set only.
- Split: by signature file (75/25), seed 42. CPU only.

Run: `pip install -r requirements.txt && python analysis.py`. Outputs `results.json` and `figures/comparison.png`.

## Reference

This project is motivated by the following paper:

> M Kazerounian, JS Foster, B Min (2021). *SimTyper: sound type inference for Ruby using type equality prediction*. Proceedings of the ACM on. https://dl.acm.org/doi/abs/10.1145/3485483

Paper author: Prof. Jeff Foster, Tufts University (jfoster@cs.tufts.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
