import numpy as np
import pandas as pd
import pytest
from provguard.leakage import LeakageIndex

def test_persisted_representation_is_batch_order_and_population_invariant(tmp_path):
    refs=["Write a security advisory about encoded payload injection and output filtering.","Explain the climate of Argentina."]
    pool=["Write a security advisory about encoded payload injection and context isolation.","Explain the climate of Argentina!","List prime numbers smaller than twenty."]
    index=LeakageIndex.build(refs,pool,tmp_path)
    candidates=pd.DataFrame({"text":pool,"candidate_id":["a","b","c"]})
    first,_=index.filter(candidates,batch=3)
    disk=LeakageIndex.load(tmp_path)
    reordered,_=disk.filter(candidates.iloc[::-1],batch=1)
    np.testing.assert_array_equal(first.sort_values("candidate_id").max_reference_similarity,reordered.sort_values("candidate_id").max_reference_similarity)
    for _,row in candidates.iterrows():
        alone,_=disk.filter(pd.DataFrame([row]))
        assert (row.candidate_id in set(first.candidate_id)) == (len(alone)==1)
    with pytest.raises(ValueError):
        disk.filter(candidates,threshold=.89)

def test_persisted_index_detects_corruption(tmp_path):
    LeakageIndex.build(["Explain prompt injection."],["Write a short story about penguins."],tmp_path)
    with (tmp_path/"references.json").open("a",encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(ValueError,match="artifact changed"):
        LeakageIndex.load(tmp_path)
