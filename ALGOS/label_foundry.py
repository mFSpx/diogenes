#!/usr/bin/env python3
"""Weak supervision labeling primitives."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable
@dataclass(frozen=True)
class LabelingFunctionResult: lf_name: str; doc_id: str; label: int
@dataclass(frozen=True)
class ProbabilisticLabel: doc_id: str; label: int; confidence: float
@dataclass(frozen=True)
class LabelError: doc_id: str; given_label: int; suggested_label: int; error_probability: float
def labeling_function(name: str|None=None):
    def deco(fn: Callable[[dict],int]): fn.lf_name=name or fn.__name__; return fn
    return deco
def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    votes=defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0,1): votes[r.doc_id].append(r.label)
    out=[]
    for d,vs in votes.items():
        if not vs: out.append(ProbabilisticLabel(d,0,0.5)); continue
        c=Counter(vs); label=1 if c[1]>=c[0] else 0; out.append(ProbabilisticLabel(d,label,c[label]/len(vs)))
    return out
def find_label_errors(docs: list[dict], given: list[int], probs: list[float], threshold: float=0.65) -> list[LabelError]:
    if not (len(docs)==len(given)==len(probs)): raise ValueError('length mismatch')
    errs=[]
    for doc,g,p in zip(docs,given,probs):
        errp=p if g==0 else 1.0-p
        if errp>=threshold: errs.append(LabelError(str(doc.get('id',len(errs))),g,1-g,errp))
    return sorted(errs,key=lambda e:-e.error_probability)
