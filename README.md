# Predicting equal parameter types in Ruby signatures

SimTyper ("SimTyper: sound type inference for Ruby using type equality prediction", 2021) relies on predicting whether two program elements have equal types. This experiment studies a related, simplified task: predicting whether two Ruby method parameters have the same declared type, from their names and context.

## Question
Can a class-balanced gradient-boosting model with richer name and context features predict whether two Ruby parameters have equal types better than a name-similarity threshold predictor, on held-out Ruby files?

## Dataset
Ruby's public RBS core and stdlib signatures (ruby/rbs on GitHub, downloaded as a zip). Parameters (name, method, class, declared type) are parsed with regexes and turned into labelled pairs (label = same declared type, fixed 50/50 balance). The split is grouped by signature file (about 75% train / 25% test), so test classes and libraries are unseen. This is not the paper's own dataset.

## Models
- **Baseline:** character n-gram TF-IDF cosine similarity of parameter names, with a threshold tuned on the training pairs. It is a stand-in for the paper's similarity-based predictor, not the paper's method.
- **Improved:** class-balanced LightGBM, tuned with grouped cross-validation on the training set. It uses token Jaccard, edit ratio, shared prefix/suffix tokens, same method/class/file, out-of-fold name-pair type-agreement priors and name frequency.

Main metric: F1 of the equal-types class. Secondary metrics: precision and ROC-AUC. The test set is used once at the end.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> M Kazerounian, JS Foster, B Min (2021). *SimTyper: sound type inference for Ruby using type equality prediction*. Proceedings of the ACM on. https://dl.acm.org/doi/abs/10.1145/3485483

Paper author: Prof. Jeff Foster, Tufts University (jfoster@cs.tufts.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
