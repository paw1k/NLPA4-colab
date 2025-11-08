from __future__ import annotations
import re
import math
import gc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional

# Only needed for entailment mode. Other modes work without these installed.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


# Small stopword list so we don't depend on external downloads.
_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","to","of","in","on","for",
    "by","with","as","at","from","that","this","these","those","it","its","is","am","are","was","were",
    "be","been","being","into","their","his","her","hers","him","he","she","they","them","we","our",
    "you","your","yours","i","me","my","mine","not","no","do","does","did","doing","over","under",
    "than","too","very","can","could","should","would","will","just","about","also"
}
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def content_words(text: str) -> List[str]:
    return [w for w in _tokenize(text) if w not in _STOPWORDS]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A = set(a)
    B = set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / float(len(A | B))


def split_passage_into_sentences(passage_text: str) -> List[str]:
    """
    Retrieved passages in your JSONL are often wrapped as <s> ... </s>.
    Prefer those tags. Fall back to a naive regex splitter if tags are absent.
    """
    if "<s>" in passage_text and "</s>" in passage_text:
        parts = [p.strip() for p in passage_text.replace("</s>", "").split("<s>") if p.strip()]
        return parts
    return [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", passage_text) if s.strip()]


@dataclass
class FactExample:
    sent: str
    passages: List[str]
    label: str  # 'S', 'NS', or 'IR'

    def gold_binary(self) -> str:
        # Map 'IR' to 'NS' for binary evaluation (supported vs not-supported/irrelevant).
        return 'S' if self.label == 'S' else 'NS'

    def __repr__(self) -> str:
        return f"FactExample(sent={self.sent!r}, label={self.label!r}, passages={len(self.passages)} passages)"


class WordRecallThresholdFactChecker:
    """
    Lexical baseline: compute max Jaccard overlap on content words between the claim
    and ANY sentence in the retrieved passages. Predict 'S' if >= threshold.
    """
    def __init__(self, threshold: float = 0.22):
        self.threshold = threshold

    def score(self, claim: str, candidate_sent: str) -> float:
        return jaccard(content_words(claim), content_words(candidate_sent))

    def predict(self, example: FactExample) -> Tuple[str, float]:
        best = 0.0
        for p in example.passages:
            for s in split_passage_into_sentences(p):
                best = max(best, self.score(example.sent, s))
        pred = 'S' if best >= self.threshold else 'NS'
        return pred, best


class EntailmentModel:
    """
    Thin wrapper around a HuggingFace NLI checkpoint (default: roberta-large-mnli).
    Provides check_entailment(premise, hypothesis) -> (p_entail, p_neutral, p_contra).
    """
    def __init__(self, model_name: str = "roberta-large-mnli", use_cuda: bool = False):
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("transformers/torch not available – cannot use entailment mode.")
        dev = torch.device("cuda") if (use_cuda and torch and torch.cuda.is_available()) else torch.device("cpu")
        self.device = dev
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # Try to detect label order; fall back to MNLI default (0:contra,1:neutral,2:entailment)
        self._label_ix = None
        id2label = getattr(self.model.config, "id2label", None)
        if id2label:
            inv = {v.lower(): k for k, v in id2label.items()}
            if "entailment" in inv and "neutral" in inv and "contradiction" in inv:
                self._label_ix = (inv["entailment"], inv["neutral"], inv["contradiction"])

    def check_entailment(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).flatten().tolist()
        if self._label_ix is None:
            p_contra, p_neutral, p_entail = probs  # MNLI default order
        else:
            p_entail = probs[self._label_ix[0]]
            p_neutral = probs[self._label_ix[1]]
            p_contra = probs[self._label_ix[2]]
        # clean up
        del inputs, logits
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return float(p_entail), float(p_neutral), float(p_contra)


class EntailmentFactChecker:
    """
    Runs NLI sentence-by-sentence over retrieved passages.
    Uses a light lexical pruning step (word-overlap) to limit NLI calls.
    """
    def __init__(
            self,
            entail_model: EntailmentModel,
            entail_threshold: float = 0.50,
            prune_model: Optional[WordRecallThresholdFactChecker] = None,
            prune_threshold: float = 0.05,
            max_candidates: Optional[int] = 12,
    ):
        self.model = entail_model
        self.entail_threshold = entail_threshold
        self.prune_model = prune_model or WordRecallThresholdFactChecker(threshold=prune_threshold)
        self.prune_threshold = prune_threshold
        self.max_candidates = max_candidates

    def _generate_candidates(self, example: FactExample) -> List[str]:
        # Score all sentences lexically and keep the top-K above prune_threshold.
        scored: List[Tuple[float, str]] = []
        for p in example.passages:
            for s in split_passage_into_sentences(p):
                sc = self.prune_model.score(example.sent, s)
                if sc >= self.prune_threshold:
                    scored.append((sc, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        if self.max_candidates is not None:
            scored = scored[: self.max_candidates]
        return [s for _, s in scored]

    def predict(self, example: FactExample) -> Tuple[str, float]:
        best_entail = 0.0
        for cand in self._generate_candidates(example):
            p_ent, _, _ = self.model.check_entailment(premise=cand, hypothesis=example.sent)
            if p_ent > best_entail:
                best_entail = p_ent
        pred = 'S' if best_entail >= self.entail_threshold else 'NS'
        return pred, best_entail
