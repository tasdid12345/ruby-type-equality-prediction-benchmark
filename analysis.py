"""Equal-type prediction for Ruby parameter pairs (RBS signatures): name-similarity baseline vs LightGBM."""
import io, json, os, re, sys, time, zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import numpy as np, requests, lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

URL = "https://codeload.github.com/ruby/rbs/zip/refs/heads/master"
SEED, N_TRAIN, N_TEST, K = 42, 40000, 15000, 5
DEF = re.compile(r'^\s*(?:(?:private|public)\s+)?def\s+(?:self(?:\?)?\.)?(\S+?)\s*:\s+(.*)$')
CLS = re.compile(r'^\s*(?:class|module|interface)\s+([A-Z][\w:]*)')
CONT = re.compile(r'^\s*\|\s*(.*)$')
NOT_NAMES = {"nil", "true", "false", "bool", "untyped", "void", "self", "instance", "class", "top", "bot"}

def download():
    for k in range(5):  # retry with backoff on 429/5xx
        try:
            r = requests.get(URL, timeout=60)
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status(); return r.content
        except requests.RequestException as e:
            print("download error", e)
        time.sleep(2 ** k)
    raise RuntimeError("download failed")

def split_top(s, sep):  # split on sep outside brackets
    out, d, cur = [], 0, ""
    for c in s:
        d += c in "([{"; d -= c in ")]}"
        if c == sep and d == 0: out.append(cur); cur = ""
        else: cur += c
    return out + [cur]

def parse_params(seg):  # [(name, type)] from "(T a, ?T b, k: T) -> R"
    seg = re.sub(r'^\s*\[[^\]]*\]\s*', '', seg)
    if not seg.startswith('('): return []
    d, end = 0, None
    for i, c in enumerate(seg):
        d += c in "([{"; d -= c in ")]}"
        if d == 0: end = i; break
    if end is None: return []
    res = []
    for p in split_top(seg[1:end], ','):
        p = re.sub(r'^(\?|\*\*|\*)\s*', '', p.strip())
        m = re.match(r'^([a-z_]\w*):(?!:)\s*(.+)$', p)
        if m: res.append((m.group(1), m.group(2)))
            continue
        m = re.match(r'^(.+)\s+([a-z_]\w*)$', p)
        if m and not m.group(1).rstrip().endswith(('|', '&')) and m.group(2) not in NOT_NAMES:
            res.append((m.group(2), m.group(1)))
    return res

def tok(n): return [t for t in re.split(r'_+', n.rstrip('?!').lower()) if t] or [n]

def load_records():
    z = zipfile.ZipFile(io.BytesIO(download())); seen, recs = set(), []
    for fn in sorted(z.namelist()):
        if not re.search(r'/(core|stdlib)/.*\.rbs$', fn): continue
        cls, meth = "", None
        for line in z.read(fn).decode("utf-8", "ignore").splitlines():
            if not line.strip() or line.strip().startswith('#'): continue
            m = CLS.match(line)
            if m: cls, meth = m.group(1), None; continue
            m = DEF.match(line)
            if m: meth, rest = m.group(1), m.group(2)
            elif meth and CONT.match(line): rest = CONT.match(line).group(1)
            else: continue
            for seg in split_top(rest, '|'):
                for name, typ in parse_params(seg):
                    typ = re.sub(r'\s+', ' ', typ).strip()
                    key = (fn, cls, meth, name, typ)
                    if typ == "untyped" or key in seen: continue  # untyped carries no label
                    seen.add(key)
                    recs.append(dict(file=fn, cls=cls, meth=meth, name=name, type=typ, tok=tok(name),
                                     mt=re.findall(r'[a-z0-9]+', meth.lower())))
    return recs

def make_pairs(recs, n, rng):  # balanced pairs: 50% same type, 50% different type
    by = defaultdict(list)
    for i, r in enumerate(recs): by[r["type"]].append(i)
    multi = [i for i, r in enumerate(recs) if len(by[r["type"]]) > 1]
    pairs, seen = [], set()
    while len(pairs) < n // 2:
        i = multi[rng.integers(len(multi))]; j = by[recs[i]["type"]][rng.integers(len(by[recs[i]["type"]]))]
        if i != j and (i, j) not in seen: seen.add((i, j)); pairs.append((i, j, 1))
    while len(pairs) < n:
        i, j = rng.integers(len(recs), size=2)
        if recs[i]["type"] != recs[j]["type"] and (i, j) not in seen: seen.add((i, j)); pairs.append((int(i), int(j), 0))
    return pairs

def build_stats(recs):  # type distributions per full name and per last name token
    n, s = defaultdict(Counter), defaultdict(Counter)
    for r in recs: n[r["name"]][r["type"]] += 1; s[r["tok"][-1]][r["type"]] += 1
    tot = lambda d: {k: sum(v.values()) for k, v in d.items()}
    return dict(n=n, s=s, nt=tot(n), st=tot(s))

def prior(d, tot, x, y):  # probability that random draws of x and y share a type
    a, b = d.get(x), d.get(y)
    if not a or not b: return -1.0
    return sum(c * b.get(t, 0) for t, c in a.items()) / (tot[x] * tot[y])

def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0

def cosine(pairs, recs, vec):
    X = vec.transform([r["name"] for r in recs])
    ia, ib = [p[0] for p in pairs], [p[1] for p in pairs]
    return np.asarray(X[ia].multiply(X[ib]).sum(axis=1)).ravel()

def featurize(pairs, recs, stats_for, vec):  # same pairs/records feed both models
    rows = []
    for i, j, _ in pairs:
        a, b, st = recs[i], recs[j], stats_for(i, j)
        ca, cb = st["nt"].get(a["name"], 0), st["nt"].get(b["name"], 0)
        rows.append([jac(a["tok"], b["tok"]), SequenceMatcher(None, a["name"], b["name"]).ratio(),
                     a["name"] == b["name"], a["tok"][-1] == b["tok"][-1], a["tok"][0] == b["tok"][0],
                     a["meth"] == b["meth"], a["cls"] == b["cls"], a["file"] == b["file"], jac(a["mt"], b["mt"]),
                     prior(st["n"], st["nt"], a["name"], b["name"]),
                     prior(st["s"], st["st"], a["tok"][-1], b["tok"][-1]),
                     np.log1p(min(ca, cb)), np.log1p(max(ca, cb))])
    return np.column_stack([cosine(pairs, recs, vec), np.array(rows, dtype=float)])

def best_thr(y, s, grid):
    return max(grid, key=lambda t: f1_score(y, s >= t))

def evaluate(y, score, thr):  # shared evaluation function
    p = score >= thr
    return dict(f1=f1_score(y, p), precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p), auc=roc_auc_score(y, score))

def main():
    recs = load_records(); print("records:", len(recs))
    if len(recs) < 2000: sys.exit("too few parameters parsed")
    groups = [r["file"] for r in recs]
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(recs, groups=groups))
    tr, te = [recs[i] for i in tr_i], [recs[i] for i in te_i]
    rng = np.random.default_rng(SEED)
    ptr, pte = make_pairs(tr, N_TRAIN, rng), make_pairs(te, N_TEST, rng)
    ytr, yte = np.array([p[2] for p in ptr]), np.array([p[2] for p in pte])
    print("train/test records", len(tr), len(te), "pairs", len(ptr), len(pte))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit([r["name"] for r in tr])
    # Baseline: TF-IDF char n-gram cosine of names, threshold tuned on training pairs
    ctr, cte = cosine(ptr, tr, vec), cosine(pte, te, vec)
    bthr = best_thr(ytr, ctr, np.unique(np.quantile(ctr, np.linspace(0.01, 0.99, 99))))
    base = evaluate(yte, cte, bthr); print("baseline", base)
    # Improved features: priors for train pairs are out-of-fold by signature file (excluding both pairs' folds)
    files = sorted({r["file"] for r in tr}); fold = {f: k % K for k, f in enumerate(rng.permutation(files))}
    cache = {}
    def stats_oof(i, j):
        key = frozenset((fold[tr[i]["file"]], fold[tr[j]["file"]]))
        if key not in cache: cache[key] = build_stats([r for r in tr if fold[r["file"]] not in key])
        return cache[key]
    full = build_stats(tr)
    Xtr, Xte = featurize(ptr, tr, stats_oof, vec), featurize(pte, te, lambda i, j: full, vec)
    gtr = np.array([fold[tr[p[0]]["file"]] for p in ptr]); print("features", Xtr.shape)
    mk = lambda cfg, seed: lgb.LGBMClassifier(num_leaves=cfg[0], n_estimators=cfg[1], learning_rate=cfg[2],
        class_weight="balanced", subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, n_jobs=4, verbose=-1)
    best = None
    for cfg in [(15, 200, 0.05), (31, 300, 0.05), (63, 300, 0.03), (15, 500, 0.03)]:  # grouped CV on train only
        oof = np.zeros(len(ytr))
        for a, b in GroupKFold(K).split(Xtr, ytr, gtr):
            oof[b] = mk(cfg, SEED).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
        auc = roc_auc_score(ytr, oof); print("cv", cfg, round(auc, 4))
        if best is None or auc > best[0]: best = (auc, cfg, oof)
    ithr = best_thr(ytr, best[2], np.linspace(0.05, 0.95, 91)); print("chosen", best[1], "thr", ithr)
    runs = []
    for seed in (42, 43, 44):
        m = mk(best[1], seed).fit(Xtr, ytr)
        runs.append(evaluate(yte, m.predict_proba(Xte)[:, 1], ithr)); print("improved seed", seed, runs[-1])
    imp = {k: float(np.mean([r[k] for r in runs])) for k in base}
    imp_std = {k: float(np.std([r[k] for r in runs])) for k in base}
    metrics = {"baseline_score": float(base["f1"]), "improved_score": imp["f1"],
               "improvement": imp["f1"] - float(base["f1"]), "baseline_score_std": 0.0,
               "improved_score_std": imp_std["f1"], "baseline_threshold": float(bthr), "improved_threshold": float(ithr),
               "cv_auc_train_best": float(best[0]), "n_records": len(recs)}
    for k in base: metrics[f"baseline_{k}"] = float(base[k]); metrics[f"improved_{k}"] = imp[k]
    os.makedirs("figures", exist_ok=True)
    ks = ["f1", "precision", "auc"]; x = np.arange(3)
    plt.bar(x - .2, [base[k] for k in ks], .4, label="name-similarity baseline")
    plt.bar(x + .2, [imp[k] for k in ks], .4, label="LightGBM")
    plt.xticks(x, ["F1", "Precision", "ROC-AUC"]); plt.legend(); plt.title("Held-out files"); plt.savefig("figures/comparison.png", dpi=120)
    res = {"question": "Can a class-balanced gradient-boosting model with richer name and context features predict whether two Ruby parameters have equal types better than a name-similarity threshold predictor, on held-out Ruby files?",
           "method": "Pairs of parameters from ruby/rbs core+stdlib signatures (50% same declared type), split by signature file; baseline TF-IDF char n-gram cosine threshold vs class-balanced LightGBM on richer features, tuned by grouped CV on train.",
           "dataset": {"name": "ruby/rbs core and stdlib signatures (parameter pairs)", "source_url": URL,
                       "n_rows": len(ptr) + len(pte), "n_train": len(ptr), "n_test": len(pte),
                       "same_as_paper": False, "baseline_is_papers_method": False},
           "metric_name": "F1 of the equal-types class", "higher_is_better": True,
           "baseline": "Character n-gram TF-IDF cosine similarity of parameter names, threshold tuned on training pairs (stand-in for SimTyper's similarity-based predictor).",
           "improved": "Class-balanced LightGBM on token Jaccard, edit ratio, shared prefix/suffix tokens, same method/class/file, method-name overlap, out-of-fold name-type agreement priors, name frequency and the cosine score.",
           "metrics": metrics,
           "notes": "Params parsed by regex from RBS (single-line defs, 'untyped' params dropped, types compared as exact strings). Pairs are sampled 50/50 within each file-grouped partition, so results depend on this synthetic balance. Train priors are out-of-fold by file; grouped CV reuses them, so CV is slightly optimistic. Baseline is deterministic (std 0); LightGBM averaged over seeds 42-44. Test set used once."}
    json.dump(res, open("results.json", "w", encoding="utf-8"), indent=2)
    if not np.isfinite(metrics["improved_score"]): sys.exit(1)

if __name__ == "__main__":
    main()
