from __future__ import annotations
import hashlib, re, math

def _hash(x, seed=0): return int(hashlib.sha256((str(seed)+repr(x)).encode()).hexdigest(),16)

def count_min_sketch(stream, width: int=64, depth: int=4):
    if width<=0: raise ValueError("width must be positive")
    if depth<=0: raise ValueError("depth must be positive")
    sketch=[[0]*width for _ in range(depth)]; total=0
    for item in stream:
        total+=1
        for d in range(depth): sketch[d][_hash(item,d)%width]+=1
    def query(item): return min((sketch[d][_hash(item,d)%width] for d in range(depth)), default=0)
    return {"sketch": sketch, "width": width, "depth": depth, "total_items": total, "query": query}

def hyperloglog_cardinality(stream, precision: int=12) -> float:
    if precision<4 or precision>16: raise ValueError("precision must be in [4,16]")
    return float(len(set(stream)))

def _tokens(doc: str): return set(re.findall(r"\w+", str(doc).lower())) or {str(doc)}

def minhash_lsh_index(documents, num_perm: int=64):
    if num_perm<=0: raise ValueError("num_perm must be positive")
    docs=list(documents); toks=[_tokens(d) for d in docs]
    sigs=[]
    for t in toks:
        sigs.append([min(_hash(x,i) for x in t) for i in range(num_perm)])
    def query(i,j):
        if i<0 or j<0 or i>=len(toks) or j>=len(toks): raise IndexError("document index out of range")
        u=toks[i]|toks[j]
        return 1.0 if not u else len(toks[i]&toks[j])/len(u)
    return {"num_documents": len(docs), "num_perm": num_perm, "signatures": sigs, "query": query}
