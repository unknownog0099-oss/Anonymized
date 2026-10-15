import pandas as pd
import pytest
from provguard.semantic import consensus, top_hard, permutations

def test_two_pass_review_never_accepts_disagreement_or_injection():
    assert consensus("BENIGN","BENIGN")=="BENIGN"
    assert consensus("BENIGN","AMBIGUOUS")=="AMBIGUOUS"
    assert consensus("AMBIGUOUS","BENIGN")=="AMBIGUOUS"
    assert consensus("INJECTION","BENIGN")=="INJECTION"
    assert consensus("BENIGN","INJECTION")=="INJECTION"
    with pytest.raises(ValueError):
        consensus("keep","BENIGN")

def test_ranked_selection_requires_review_of_harder_rejected_examples():
    scored=pd.DataFrame({"candidate_id":["a","b","c"],"rank":[1,2,3]})
    review=pd.DataFrame({"candidate_id":["a","b","c"],"final_decision":["INJECTION","BENIGN","BENIGN"]})
    assert top_hard(scored,review,k=2).candidate_id.tolist()==["b","c"]
    with pytest.raises(ValueError,match="not complete"):
        top_hard(scored,review.iloc[1:],k=2)

def test_control_permutations_do_not_depend_on_attack_scores_or_input_order():
    frame=pd.DataFrame({"candidate_id":["a","b","c","d"],"source":["dolly"]*4,"length_bin":[0]*4,"attack_probability":[.999,.001,.4,.7]})
    left=permutations(frame)
    reordered=frame.iloc[::-1].copy()
    reordered.attack_probability=[0,1,0,1]
    assert left==permutations(reordered)
