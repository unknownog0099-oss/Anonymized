"""Join independent full-text semantic decisions and construct controlled sets."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .core import DATA, ROOT, save_json, sha
from .leakage import LeakageIndex

LABELS={"BENIGN","INJECTION","AMBIGUOUS"}
BATCHES=DATA/"semantic_batches"

def consensus(a,b):
    if a not in LABELS or b not in LABELS:
        raise ValueError("Unknown semantic label.")
    if "INJECTION" in (a,b):
        return "INJECTION"
    return "BENIGN" if a==b=="BENIGN" else "AMBIGUOUS"

def reuse_scores():
    canonical=pd.read_csv(DATA/"clean_candidates.csv",keep_default_na=False)
    old=pd.read_csv(DATA/"archive/pre_repair_20261013/scored_candidates.csv",keep_default_na=False)
    keep=["candidate_id","text","attack_probability","baseline_seed",*[f"attack_probability_seed_{s}" for s in [13,42,123]]]
    result=canonical.merge(old[keep],on="candidate_id",suffixes=("","_scored"),validate="one_to_one")
    if len(result)!=len(canonical) or not result.text.eq(result.text_scored).all():
        raise ValueError("Cached scores are incomplete or text changed; score only missing candidates.")
    result=result.drop(columns="text_scored")
    for s in [13,42,123]:
        result[f"score_seed{s}"]=result[f"attack_probability_seed_{s}"]
    mean=result[[f"score_seed{s}" for s in [13,42,123]]].mean(axis=1)
    if not np.allclose(mean,result.attack_probability,rtol=0,atol=1e-15):
        raise ValueError("Cached mean score does not match the three selectors.")
    result["mean_attack_probability"]=mean
    result=result.sort_values(["mean_attack_probability","candidate_id"],ascending=[False,True]).reset_index(drop=True)
    result["rank"]=np.arange(1,len(result)+1)
    # Length quartiles depend on text/ID only, not hardness or review order.
    ordered=result.sort_values(["char_length","candidate_id"])
    bins=pd.Series(np.minimum(np.arange(len(ordered))*4//len(ordered),3),index=ordered.candidate_id)
    result["length_bin"]=result.candidate_id.map(bins)
    result.to_csv(DATA/"scored_candidates.csv",index=False)
    save_json(ROOT/"reports/selector_score_provenance.json",{"selector_seeds":[13,42,123],"method":"exact-ID and exact-text verified reuse of saved unaugmented-selector probabilities", "old_scores_sha256":sha(DATA/"archive/pre_repair_20261013/scored_candidates.csv"),"canonical_scored_sha256":sha(DATA/"scored_candidates.csv"),"candidate_n":len(result),"new_inference_n":0})
    return result

def decisions(pass_name):
    files=list(BATCHES.glob(f"pass_{pass_name}*.jsonl"))
    parts=[pd.read_json(p,lines=True,dtype={"candidate_id":str}) for p in files if p.stat().st_size]
    if not parts:
        return pd.DataFrame(columns=["candidate_id","decision","reason"])
    df=pd.concat(parts,ignore_index=True)
    if df.candidate_id.duplicated().any() or not set(df.decision).issubset(LABELS) or df.reason.map(lambda x:not isinstance(x,str) or not x.strip()).any():
        raise ValueError(f"Invalid/duplicate semantic pass {pass_name} decisions.")
    return df

def merge_review(scored=None):
    scored=pd.read_csv(DATA/"scored_candidates.csv",keep_default_na=False) if scored is None else scored
    inputs=[pd.read_json(p,lines=True,dtype={'candidate_id':str}) for p in BATCHES.glob('*.jsonl') if not p.name.startswith('pass_')]
    originals=pd.concat(inputs,ignore_index=True)
    if originals.candidate_id.duplicated().any():
        raise ValueError('Review input candidate IDs repeat across batches.')
    bound=originals[['candidate_id','text']].merge(scored[['candidate_id','text']],on='candidate_id',suffixes=('_input','_scored'),validate='one_to_one')
    if len(bound)!=len(originals) or not bound.text_input.eq(bound.text_scored).all():
        raise ValueError('Review input text differs from canonical scored candidates.')
    a=decisions("A").rename(columns={"decision":"review_A","reason":"reason_A"})
    b=decisions("B").rename(columns={"decision":"review_B","reason":"reason_B"})
    input_ids=set(originals.candidate_id)
    if not set(a.candidate_id).issubset(input_ids) or not set(b.candidate_id).issubset(input_ids):
        raise ValueError('Adjudication decision not bound to a recorded full-text input.')
    joined=scored.merge(a,on="candidate_id",validate="one_to_one").merge(b,on="candidate_id",validate="one_to_one")
    joined["final_decision"]=[consensus(a,b) for a,b in zip(joined.review_A,joined.review_B)]
    joined["adjudication_method"]="two independent Codex review instances; full text; scores withheld; no human reviewer"
    joined.to_csv(DATA/"semantic_review.csv",index=False)
    save_json(ROOT/"reports/semantic_adjudication_audit.json",{"method":joined.adjudication_method.iloc[0] if len(joined) else "independent automated semantic adjudication", "pass_A_n":len(a),"pass_B_n":len(b),"paired_review_n":len(joined),"final_counts":joined.final_decision.value_counts().to_dict(),"disagreement_n":int((joined.review_A!=joined.review_B).sum()),"input_batches":{p.name:sha(p) for p in BATCHES.glob("*.jsonl") if not p.name.startswith("pass_")},"decision_artifacts":{p.name:sha(p) for p in BATCHES.glob("pass_*.jsonl")},"review_sha256":sha(DATA/"semantic_review.csv"),"procedure_artifacts":{p.name:sha(p) for p in BATCHES.glob("*_procedure.md")},"model_version":"Exact serving model version not exposed; independent Codex instances with separate histories/prompts", "independence_limitation":"Separate review instantiations can share correlated model errors; agreement is not a guarantee of valid labels.","limitation":"Automated model adjudication is fallible and not human annotation; independent human validation remains useful before publication."})
    return joined

def top_hard(scored,review,k=1000):
    labels=review.set_index("candidate_id").final_decision.to_dict()
    ids=[]
    for cid in scored.sort_values("rank").candidate_id:
        if cid not in labels:
            raise ValueError(f"Rank-order adjudication not complete at candidate {cid}.")
        if labels[cid]=="BENIGN":
            ids.append(cid)
        if len(ids)==k:
            break
    if len(ids)!=k:
        raise ValueError(f"Only {len(ids)} independently approved benign candidates, need {k}.")
    return review.set_index("candidate_id").loc[ids].reset_index()

def permutations(scored,seed=42):
    rng=np.random.default_rng(seed)
    orders={}
    for key,group in scored.groupby(["source","length_bin"],sort=True):
        ids=sorted(group.candidate_id.tolist())
        orders[key]=np.asarray(ids)[rng.permutation(len(ids))].tolist()
    return orders

def control_review_queue(provisional=False):
    scored=pd.read_csv(DATA/"scored_candidates.csv",keep_default_na=False)
    if provisional:
        a=decisions("A")
        proxy=scored.merge(a,on="candidate_id",validate="one_to_one")
        proxy["final_decision"]=proxy.decision
        hard=top_hard(scored,proxy)
    else:
        hard=top_hard(scored,merge_review(scored))
    orders=permutations(scored)
    wanted=[]
    for key,group in hard.groupby(["source","length_bin"]):
        # This is a predetermined review oversample, not resampling on results.
        wanted+=orders[key][:min(len(orders[key]),len(group)+40)]
    previously=set(pd.read_json(BATCHES/"ranked_top_2000.jsonl",lines=True).candidate_id)
    queue=scored.set_index("candidate_id").loc[[cid for cid in wanted if cid not in previously]].reset_index()
    queue[["candidate_id","text","source"]].to_json(BATCHES/"random_control_review.jsonl",orient="records",lines=True,force_ascii=False)
    save_json(ROOT/"reports/random_control_review_plan.json",{"seed":42,"pool":"all canonical leakage-filtered real candidates", "selection":"fixed random permutation within source/full-pool length quartile; take first semantically approved candidates without a hardness condition", "quota_is_provisional":provisional,"target_counts":{f"{s}:{b}":len(g) for (s,b),g in hard.groupby(["source","length_bin"])},"queue_n":len(queue),"new_input_sha256":sha(BATCHES/"random_control_review.jsonl")})
    return queue

def build_final_sets():
    scored=pd.read_csv(DATA/"scored_candidates.csv",keep_default_na=False)
    review=merge_review(scored)
    hard=top_hard(scored,review)
    labels=review.set_index("candidate_id").final_decision.to_dict()
    orders=permutations(scored)
    chosen=[]
    missing=[]
    for key,group in hard.groupby(["source","length_bin"]):
        accepted=[]
        for cid in orders[key]:
            if cid not in labels:
                missing.append(cid)
                break
            if labels[cid]=="BENIGN":
                accepted.append(cid)
            if len(accepted)==len(group):
                break
        if len(accepted)!=len(group):
            if not missing:
                raise ValueError(f"Insufficient semantically benign examples in random stratum {key}.")
        chosen+=accepted
    if missing:
        raise ValueError(f"Random control review incomplete in fixed permutation: {missing}")
    random=review.set_index("candidate_id").loc[chosen].reset_index()
    if len(random)!=1000:
        raise AssertionError("Random control must have exactly 1000 rows.")
    verifier=LeakageIndex.load()
    for name,frame in [(f"provguard_{k}",hard.iloc[:k].copy()) for k in [250,500,1000]]+[("random_real_1000",random)]:
        checked,log=verifier.filter(frame)
        if len(checked)!=len(frame):
            raise ValueError(f"Canonical independent leakage failure for {name}: {log}")
        checked["label"]=0
        checked["decision"]="keep"
        checked["reason"]=checked.reason_A+" | "+checked.reason_B
        checked["reviewer"]="automated semantic adjudication A+B"
        checked.to_csv(DATA/f"{name}.csv",index=False)
    if not hard.groupby(["source","length_bin"]).size().equals(random.groupby(["source","length_bin"]).size()):
        raise AssertionError("Random source/length strata differ from treatment.")
    stats=[]
    for name,frame in [("provguard_1000",hard),("random_real_1000",random)]:
        x=frame.mean_attack_probability
        stats.append({"method":name,"n":len(frame),"min":x.min(),"q1":x.quantile(.25),"median":x.median(),"mean":x.mean(),"q3":x.quantile(.75),"max":x.max(),**{f"n_above_{t}":int((x>t).sum()) for t in [.5,.25,.1,.05,.01]}})
    pd.DataFrame(stats).to_csv(ROOT/"reports/HARDNESS_ANALYSIS.csv",index=False)
    save_json(ROOT/"reports/candidate_selection_audit.json",{"primary_k":1000,"nested_k":[250,500,1000],"selection":"descending mean baseline score after paired semantic adjudication", "selector_seeds":[13,42,123],"rank_complete_through":int(hard['rank'].max()),"paired_review_n":len(review),"selected_source_counts":hard.source.value_counts().to_dict(),"random_source_counts":random.source.value_counts().to_dict(),"matched_strata":{f"{s}:{b}":len(g) for (s,b),g in hard.groupby(["source","length_bin"])},"random_seed":42,"random_overlap_n":len(set(hard.candidate_id)&set(random.candidate_id)),"augmentation_sha256":{f"{name}.csv":sha(DATA/f"{name}.csv") for name in ['provguard_250','provguard_500','provguard_1000','random_real_1000','curated_hn']},"canonical_index_corpus_sha256":verifier.metadata['canonical_corpus_sha256'],"semantic_review_sha256":sha(DATA/'semantic_review.csv'),"hardness":stats})
    print(pd.DataFrame(stats).to_string(index=False))
