"""公开包的科学配置、来源和可移植入口检查。"""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.e0.e0formal.contract import load_frozen_contract, FormalExecutionLocked
from experiments.e0.e0formal.authorization import FormalProcessSession
from experiments.e0.e0formal.identity import iter_instance_ids, IdentityError
from experiments.e0.e0formal.statistics import paired_bootstrap_difference_of_medians


def test_frozen_counts():
    cfg=load_frozen_contract()
    assert [len(list(iter_instance_ids(cfg,k))) for k in ['linear','sequential','nonlinear']]==[13500,1600,200]


def test_mutation_is_rejected():
    cfg=load_frozen_contract(); session=FormalProcessSession(cfg)
    cfg.config['formal_contract']['randomness']['master_seed']+=1
    with pytest.raises(FormalExecutionLocked): session.verify(cfg,'v1.4:E0:L:s00:r000')


def test_out_of_range_is_rejected():
    cfg=load_frozen_contract(); session=FormalProcessSession(cfg)
    with pytest.raises(IdentityError): session.verify(cfg,'v1.4:E0:L:s27:r000')


def test_public_session_does_not_mix_contracts():
    cfg=load_frozen_contract(); other=load_frozen_contract()
    with pytest.raises(FormalExecutionLocked): FormalProcessSession(cfg).verify(other,'v1.4:E0:N:g000')


def test_difference_of_medians_bootstrap():
    outcome=paired_bootstrap_difference_of_medians([1,2,3],[.5,1,1.5],repeats=100,seed=91027)
    assert outcome['observed_difference']==-1
    assert outcome['sample_count']==3


@pytest.mark.parametrize('kind,row,expected',[
    ('linear',{'status':'PASS','main_status':'PASS'},[]),
    ('linear',{'status':'LOFO_NUMERICAL_FAILURE','main_status':'PASS'},['status']),
    ('nonlinear',{'common_valid':True,'lofo_valid':False},['lofo_valid']),
    ('nonlinear',{'common_valid':True,'lofo_valid':True},[]),
    ('sequential',{'status':'NUMERICAL_FAILURE'},['status']),
])
def test_public_runner_preserves_numerical_failures(kind,row,expected):
    from scripts.run_e0 import numerical_issues
    assert numerical_issues(kind,row)==expected
