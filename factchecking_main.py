from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from typing import Dict, List, Tuple

from factcheck import (
    FactExample,
    WordRecallThresholdFactChecker,
    EntailmentModel,
    EntailmentFactChecker,
)

# ---------- Data loading ----------

def load_labels(labels_path: str) -> List[Tuple[str, str]]:
    """
    Returns list of (sent, gold_label).
    The JSONL has: { input, output, topic, cat, annotations:[ {human-atomic-facts:[{text, label}, ...]} ] }
    Labels: 'S', 'NS', 'IR'. We'll map 'IR' -> 'NS' during evaluation (binary).
    """
    rows = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            for ann in o.get("annotations", []):
                for fact in ann.get("human-atomic-facts", []):
                    rows.append((fact["text"], fact["label"]))
    return rows


def load_passages(passages_path: str) -> Dict[str, List[str]]:
    """
    Returns dict: sent -> list of passage texts (each passage is a string, often with <s> tags around sentences).
    JSONL item: { "name": ..., "sent": "...", "passages": [ { "title": ..., "text": "..." }, ... ] }
    """
    mp: Dict[str, List[str]] = {}
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            sent = o["sent"]
            plist = [p["text"] for p in o.get("passages", [])]
            mp[sent] = plist
    return mp


def build_examples(labels_path: str, passages_path: str) -> List[FactExample]:
    lbls = load_labels(labels_path)
    psgs = load_passages(passages_path)
    examples: List[FactExample] = []
    missing = 0
    for sent, lab in lbls:
        plist = psgs.get(sent, [])
        if not plist:
            missing += 1
        examples.append(FactExample(sent=sent, passages=plist, label=lab))
    if missing:
        print(f"[warn] {missing} facts had no retrieved passages", file=sys.stderr)
    return examples


# ---------- Evaluation (binary: S vs NS) ----------

def evaluate_binary(y_true: List[str], y_pred: List[str]) -> str:
    """
    Prints accuracy and per-class P/R/F1 for classes S and NS.
    y_true should already have 'IR' mapped to 'NS'.
    """
    assert len(y_true) == len(y_pred)
    total = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if (t == 'S' and p == 'S') or (t != 'S' and p != 'S')) / total
    lines = [f"Accuracy: {acc:.4f}  (over {total} facts)"]

    def prf_for(cls: str):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return prec, rec, f1, tp, fp, fn

    for cls in ["S", "NS"]:
        prec, rec, f1, tp, fp, fn = prf_for(cls)
        lines.append(f"{cls:>3}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  tp={tp} fp={fp} fn={fn}")
    return "\n".join(lines)


# ---------- Runners ----------

def run_mode(mode: str,
             examples: List[FactExample],
             use_cuda: bool = False,
             wo_thresh: float = 0.22,
             entail_thresh: float = 0.50,
             prune_thresh: float = 0.05) -> Tuple[List[str], List[float]]:
    preds: List[str] = []
    scores: List[float] = []

    if mode == "random":
        for ex in examples:
            preds.append(random.choice(["S", "NS"]))
            scores.append(0.5)
        return preds, scores

    if mode == "always_entail":
        for _ in examples:
            preds.append("S")
            scores.append(1.0)
        return preds, scores

    if mode == "word_overlap":
        model = WordRecallThresholdFactChecker(threshold=wo_thresh)
        for ex in examples:
            pred, sc = model.predict(ex)
            preds.append(pred)
            scores.append(sc)
        return preds, scores

    if mode == "entailment":
        entail = EntailmentModel(use_cuda=use_cuda)
        checker = EntailmentFactChecker(
            entail_model=entail,
            entail_threshold=entail_thresh,
            prune_model=WordRecallThresholdFactChecker(threshold=prune_thresh),
            prune_threshold=prune_thresh,
        )
        for ex in examples:
            pred, sc = checker.predict(ex)
            preds.append(pred)
            scores.append(sc)
        return preds, scores

    raise ValueError(f"Unknown mode: {mode}")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Fact checking assignment runner")
    ap.add_argument("--mode", choices=["random", "always_entail", "word_overlap", "entailment"], required=True)
    ap.add_argument("--labels_path", default="dev_labeled_ChatGPT.jsonl")
    ap.add_argument("--passages_path", default="passages_bm25_ChatGPT_humfacts.jsonl")
    ap.add_argument("--cuda", action="store_true", help="Use CUDA for entailment model if available")
    ap.add_argument("--wo_thresh", type=float, default=0.22, help="Word-overlap decision threshold")
    ap.add_argument("--entail_thresh", type=float, default=0.50, help="Entailment decision threshold")
    ap.add_argument("--prune_thresh", type=float, default=0.05, help="Lexical pruning threshold before NLI")
    args = ap.parse_args()

    examples = build_examples(args.labels_path, args.passages_path)
    y_true = [ex.gold_binary() for ex in examples]  # Map IR -> NS

    preds, _ = run_mode(
        mode=args.mode,
        examples=examples,
        use_cuda=args.cuda,
        wo_thresh=args.wo_thresh,
        entail_thresh=args.entail_thresh,
        prune_thresh=args.prune_thresh,
    )

    print(evaluate_binary(y_true, preds))


if __name__ == "__main__":
    main()
