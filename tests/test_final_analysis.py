import numpy as np
import pandas as pd
import pytest
from provguard.final_analysis import align, crossed_bootstrap, paired_statistics, transition

def example(probs):
    return pd.DataFrame({'eval_id':['a','b','c','d'],'text':['a text','b text','c text','d text'],'true_label':[0]*4,'probability_score':probs})

def test_exact_pairing_and_positive_improvement():
    a=example([.8,.9,.1,.2])
    b=example([.2,.9,.1,.6]).iloc[::-1]
    record,delta=paired_statistics(a,b)
    assert record['fp_fixed']==record['new_fp']==1
    assert record['delta_FPR']==0
    assert record['p_value']==1
    np.testing.assert_array_equal(delta,[1,0,0,-1])
    b.loc[b.eval_id.eq('a'),'text']='changed prompt'
    with pytest.raises(ValueError,match='text/labels'):
        align(a,b)

def test_crossed_bootstrap_does_not_pool_shared_prompts():
    delta=np.array([[1,0,-1,1]])
    # Repeating a seed with identical predictions must not narrow the prompt CI.
    one=crossed_bootstrap(delta,seed=77)
    repeated=crossed_bootstrap(np.repeat(delta,5,axis=0),seed=77)
    np.testing.assert_array_equal(one,repeated)
    np.testing.assert_array_equal(crossed_bootstrap(np.ones((5,808))),[1,1])

def test_error_transitions_include_remaining_false_negatives():
    assert transition(0,1,0)=='FP_FIXED'
    assert transition(0,0,1)=='NEW_FP'
    assert transition(0,1,1)=='FP_REMAINS'
    assert transition(1,0,1)=='FN_FIXED'
    assert transition(1,1,0)=='NEW_FN'
    assert transition(1,0,0)=='FN_REMAINS'
    assert transition(1,1,1)=='UNCHANGED_CORRECT'

def test_interim_report_after_twenty_primary_runs_does_not_require_final_stats(tmp_path,monkeypatch):
    import provguard.final_analysis as module
    from provguard.study import PRIMARY,SEEDS
    report_dir=tmp_path/'reports'
    report_dir.mkdir()
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'REPORT',report_dir)
    monkeypatch.setattr(module,'DATA',tmp_path/'data')
    rows=[{'method':method,'seed':seed,'augmentation_size':0 if method=='baseline' else 1000,**{m:.8 for m in module.METRICS}} for method in PRIMARY for seed in SEEDS]
    fixture=pd.DataFrame(rows)
    monkeypatch.setattr(module,'collect',lambda:fixture)
    module.export_results(final=False)
    text=(report_dir/'FINAL_EXPERIMENT_REPORT.md').read_text(encoding='utf-8')
    assert '20/30' in text and 'Status: INCOMPLETE' in text
    assert 'H1: INCONCLUSIVE' in text
