from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str; doc_id: str; label: int
@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str; label: int; confidence: float
@dataclass(frozen=True)
class LabelError:
    doc_id: str; given_label: int; suggested_label: int; error_probability: float

def labeling_function(name: str | None = None):
    def deco(fn):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    if not batches: return []
    votes=defaultdict(list)
    order=[]
    for batch in batches:
        for r in batch:
            if r.doc_id not in votes: order.append(r.doc_id)
            if r.label in (0,1): votes[r.doc_id].append(r.label)
            else: votes.setdefault(r.doc_id, votes[r.doc_id])
    out=[]
    for d in order:
        vs=votes[d]
        if not vs: out.append(ProbabilisticLabel(d,0,0.5)); continue
        ones=sum(vs); zeros=len(vs)-ones
        lab=1 if ones>=zeros else 0
        conf=max(ones,zeros)/len(vs)
        out.append(ProbabilisticLabel(d,lab,float(conf)))
    return out

def find_label_errors(docs: list[dict], given: list[int], probs: list[float], threshold: float=0.65) -> list[LabelError]:
    if not (len(docs)==len(given)==len(probs)): raise ValueError("mismatched lengths")
    out=[]
    for doc,g,p in zip(docs,given,probs):
        pred=1 if p>=0.5 else 0
        errprob=p if g==0 else 1-p
        if pred != g and errprob >= threshold:
            out.append(LabelError(str(doc.get('id', len(out))), int(g), 1-int(g), float(errprob)))
    return sorted(out, key=lambda e:e.error_probability, reverse=True)
