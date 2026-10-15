"""A persisted, batch-invariant lexical leakage representation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from .core import ROOT, BENCH, SETS, DATA, integrity, normalize, read_texts, save_json, sha

INDEX = DATA / "leakage_index_v2"

def corpus_hash(texts):
    return hashlib.sha256(json.dumps(sorted(set(texts)), ensure_ascii=False).encode("utf-8")).hexdigest()

class LeakageIndex:
    def __init__(self, vectorizer, reference_matrix, refs, metadata):
        self.vectorizer = vectorizer
        self.reference_matrix = reference_matrix
        self.refs = set(refs)
        self.metadata = metadata

    @classmethod
    def build(cls, references, candidates, directory):
        directory = Path(directory)
        if (directory / "metadata.json").exists():
            raise FileExistsError("Canonical representation already exists; load it, never refit per arm.")
        refs = sorted({normalize(x) for x in references if normalize(x)})
        all_texts = sorted(set(refs) | {normalize(x) for x in candidates if normalize(x)})
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), dtype=np.float32)
        vectorizer.fit(all_texts)
        matrix = vectorizer.transform(refs).T.tocsr()
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, directory / "vectorizer.joblib")
        save_npz(directory / "references.npz", matrix)
        save_json(directory / "references.json", refs)
        metadata = {"version":2,"representation":"TF-IDF char_wb 3-5, L2 normalized, float32", "threshold":.90, "reference_n":len(refs),"canonical_corpus_n":len(all_texts),"reference_text_sha256":corpus_hash(refs),"canonical_corpus_sha256":corpus_hash(all_texts),"artifacts":{name:sha(directory/name) for name in ["vectorizer.joblib","references.npz","references.json"]}}
        save_json(directory / "metadata.json", metadata)
        return cls(vectorizer,matrix,refs,metadata)

    @classmethod
    def load(cls,directory=INDEX,verify=True):
        directory=Path(directory)
        metadata=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
        if verify:
            for name,expected in metadata["artifacts"].items():
                if sha(directory/name)!=expected:
                    raise ValueError(f"Canonical leakage artifact changed: {name}")
        return cls(joblib.load(directory/"vectorizer.joblib"),load_npz(directory/"references.npz"),json.loads((directory/"references.json").read_text(encoding="utf-8")),metadata)

    def filter(self,candidates,threshold=.90,batch=128):
        if threshold != .90:
            raise ValueError("Canonical protocol requires threshold 0.90.")
        normalized=candidates.text.map(normalize)
        exact=normalized.isin(self.refs)|normalized.duplicated()|normalized.eq("")
        remaining=candidates.loc[~exact].copy().reset_index(drop=True)
        scores=[]
        for start in range(0,len(remaining),batch):
            query=self.vectorizer.transform(remaining.text.iloc[start:start+batch].map(normalize))
            scores.extend((query@self.reference_matrix).max(axis=1).toarray().ravel().tolist())
        remaining["max_reference_similarity"]=scores
        keep=remaining.max_reference_similarity < threshold
        report={"before_n":len(candidates),"exact_removed":int(exact.sum()),"near_removed":int((~keep).sum()),"final_n":int(keep.sum()),"threshold":threshold,"index_corpus_sha256":self.metadata["canonical_corpus_sha256"]}
        return remaining.loc[keep].reset_index(drop=True),report

def prepare():
    frozen=integrity()
    refs=[text for rel in SETS.values() for text in read_texts(BENCH/rel).text]
    real=pd.read_csv(DATA/"real_benign_candidates.csv",keep_default_na=False)
    curated=pd.read_csv(DATA/"curated_builder/hard_negative_train.csv",keep_default_na=False)
    # Include every potential external and released curated-builder train row;
    # no evaluation labels/model errors enter this representation.
    if not (INDEX/"metadata.json").exists():
        index=LeakageIndex.build(refs,real.text.tolist()+curated.text.tolist(),INDEX)
    else:
        index=LeakageIndex.load()
    if index.metadata["reference_text_sha256"]!=corpus_hash([normalize(x) for x in refs]):
        raise ValueError("Frozen references differ from canonical index.")
    if index.metadata['canonical_corpus_sha256'] != corpus_hash([normalize(x) for x in refs+real.text.tolist()+curated.text.tolist()]):
        raise ValueError('Potential candidate corpus changed after the canonical representation was frozen.')
    cleaned,real_log=index.filter(real)
    valid_curated,curated_log=index.filter(curated)
    cleaned.to_csv(DATA/"clean_candidates.csv",index=False)
    valid_curated.to_csv(DATA/"curated_hn.csv",index=False)
    real_log.update(references=frozen,input_sha256=sha(DATA/"real_benign_candidates.csv"),output_sha256=sha(DATA/"clean_candidates.csv"))
    save_json(ROOT/"reports/leakage_filter.json",real_log)
    save_json(ROOT/"reports/canonical_leakage_audit.json",{"index":index.metadata,"real":real_log,"curated":curated_log,"benchmark_integrity":frozen})
    # A fresh disk load is the independent verification used by training.
    verifier=LeakageIndex.load()
    for df in [cleaned,valid_curated]:
        verified,_=verifier.filter(df,batch=31)
        if len(verified)!=len(df) or not np.array_equal(verified.max_reference_similarity.to_numpy(),df.max_reference_similarity.to_numpy()):
            raise AssertionError("Construction/verification decisions or scores differ.")
    print(json.dumps({"real":real_log,"curated":curated_log,"independent_verification":"passed"},indent=2))
