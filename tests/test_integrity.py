import pandas as pd
import pytest
from provguard.core import read_texts, integrity
from provguard.data import leakage_filter, audited_pool, matched_random
from provguard.analysis import paired

def test_missing_texts_never_become_prompts(tmp_path):
    path = tmp_path / "texts.csv"
    pd.DataFrame({"text": [None, "", "  ", "nan", "Explain security"], "label": [0]*5}).to_csv(path, index=False)
    df = read_texts(path)
    assert df.text.tolist() == ["Explain security"]
    assert df.eval_id.tolist() == ["texts:4"]
    with pytest.raises(ValueError):
        read_texts(path, strict=True)

def test_exact_and_near_leakage_filter_preserves_text(tmp_path):
    text = "Explain prompt injection in security documentation for engineers."
    df = pd.DataFrame({"text": ["  EXPLAIN  prompt injection in security documentation for engineers.  ", "Explain prompt injection in security documentation for engineers!", "Write a story about a small penguin."], "candidate_id": ["a", "b", "c"]})
    from provguard.leakage import LeakageIndex
    index=LeakageIndex.build([text],df.text.tolist(),tmp_path/'index')
    clean, log = leakage_filter(df, [text],index=index)
    assert clean.text.tolist() == ["Write a story about a small penguin."]
    assert log["exact_removed"] == 1
    assert log["near_removed"] == 1
    with pytest.raises(ValueError,match='References differ'):
        leakage_filter(df,['a different reference corpus'],index=index)

def test_audit_cannot_change_candidate_text():
    scored = pd.DataFrame({"candidate_id": ["a"], "text": ["Explain photosynthesis."]})
    audit = pd.DataFrame({"candidate_id": ["a"], "text": ["Different text"], "decision": ["keep"], "reason": ["ordinary request"], "reviewer": ["test"]})
    with pytest.raises(ValueError, match="does not match"):
        audited_pool(scored, audit)

def test_random_control_matches_source_length_strata():
    pool = pd.DataFrame({"candidate_id": [str(i) for i in range(80)], "char_length": list(range(20,100)), "source": ["dolly", "oasst1"]*40})
    hard = pool.iloc[::4].copy()
    random = matched_random(pool, hard, seed=42)
    bins = pd.qcut(pool.char_length.rank(method="first"), 4, labels=False)
    indexed = pool.assign(length_bin=bins).set_index("candidate_id")
    left = indexed.loc[hard.candidate_id].groupby(["source", "length_bin"]).size()
    right = indexed.loc[random.candidate_id].groupby(["source", "length_bin"]).size()
    pd.testing.assert_series_equal(left, right)
    assert random.candidate_id.is_unique

def test_paired_statistics_use_ids_and_reject_missing_rows():
    b = pd.DataFrame({"eval_id": ["a", "b", "c", "d"], "true_label": [0]*4, "probability_score": [.9,.8,.7,.1]})
    p = b.copy()
    p.probability_score = [.1,.2,.7,.9]
    stats = paired(b, p.iloc[::-1])
    assert stats["baseline_fp_fixed"] == 2
    assert stats["new_fp"] == 1
    assert stats["fpr_delta"] == -.25
    with pytest.raises(ValueError):
        paired(b, p.iloc[:3])

def test_actual_frozen_benchmark_checksums(tmp_path,monkeypatch):
    # Keep the published audit snapshot unchanged while testing its writer.
    import provguard.core as core
    monkeypatch.setattr(core,'ROOT',tmp_path)
    report = integrity()
    assert report["hard_benign"]["usable_text_n"] == 808
    assert report["hard_benign"]["missing_text_n"] == 664
