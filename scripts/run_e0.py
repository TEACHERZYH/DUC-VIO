"""默认仅运行最小样本；完整复现必须显式选择full。"""
from pathlib import Path
import os
for variable in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable]='1'
import argparse
import hashlib
import json
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.e0.e0formal.contract import load_frozen_contract
from experiments.e0.e0formal.authorization import FormalProcessSession
from experiments.e0.e0formal.identity import iter_instance_ids
from experiments.e0.e0formal.linear import run_linear_graph
from experiments.e0.e0formal.nonlinear import run_nonlinear_graph
from experiments.e0.e0formal.sequential import run_sequential_instance


def numerical_issues(kind,row):
    """只判断数值执行是否有效，不把效果不佳判为程序错误。"""
    if kind=='nonlinear':
        return [name for name in ['common_valid','lofo_valid'] if row.get(name) is not True]
    issues=[]
    if row.get('status')!='PASS': issues.append('status')
    if kind=='linear' and row.get('main_status')!='PASS': issues.append('main_status')
    return issues


def run(output,profile,kinds,shard_index=0,shards=1):
    if profile not in ('smoke','full'):
        raise ValueError('profile必须为smoke或full。')
    if not kinds or len(set(kinds))!=len(kinds) or set(kinds)-{'linear','nonlinear','sequential'}:
        raise ValueError('实验类别必须非空、有效且不重复。')
    if type(shards) is not int or type(shard_index) is not int:
        raise ValueError('分片数和编号必须为整数。')
    if shards<1 or not 0<=shard_index<shards:
        raise ValueError('分片编号必须处于[0,分片数)内。')
    if profile=='smoke' and shards!=1:
        raise ValueError('最小样本不分片。')
    contract=load_frozen_contract(); session=FormalProcessSession(contract)
    selected={kind:[identity for ordinal,identity in enumerate(iter_instance_ids(contract,kind))
                    if ordinal % shards==shard_index] for kind in kinds}
    if any(not identities for identities in selected.values()):
        raise ValueError('所选分片至少有一类实验没有实例。')
    if profile=='smoke': selected={kind:identities[:1] for kind,identities in selected.items()}
    output.mkdir(parents=True,exist_ok=False)
    counts=dict.fromkeys(kinds,0); hashes={}; failures=[]
    receipt={'profile':profile,'counts':counts,'expected_counts':{k:len(v) for k,v in selected.items()},
             'shard_index':shard_index,'shards':shards,'files_sha256':hashes,
             'python':platform.python_version(),'config_payload_sha256':contract.payload_sha256,
             'replaces_archived_paper_results':False,'failures':failures,'execution_status':'RUNNING'}
    (output/'run_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        for kind,identities in selected.items():
            with (output/(kind+'.jsonl')).open('x',encoding='utf-8',newline='\n') as stream:
                for identity in identities:
                    if kind=='linear':
                        kwargs={'holdout_samples':16,'lofo_enabled':False} if profile=='smoke' else {'process_session':session}
                        row=run_linear_graph(contract,identity.setting_id,identity.instance_id,**kwargs)
                    elif kind=='sequential':
                        kwargs={'length':25} if profile=='smoke' else {'process_session':session}
                        row=run_sequential_instance(contract,identity.setting_id,identity.instance_id,**kwargs)
                    else:
                        kwargs={'holdout_samples':16,'lofo_factors':1} if profile=='smoke' else {'process_session':session}
                        row=run_nonlinear_graph(contract,identity.instance_id,**kwargs)
                    if row.get('stable_id')!=identity.stable_id:
                        raise ValueError('数值内核返回的实例身份不匹配。')
                    row['kernel_provenance']=row['provenance']
                    row['provenance']='public_reproduction_e0' if profile=='full' else 'dev_toy'
                    row['science_claim_eligible']=False
                    row['evidence_eligibility']='INDEPENDENT_REPRODUCTION_REQUIRES_REVIEW' if profile=='full' else 'SMOKE_ONLY'
                    row['formal_execution_unlock']=False
                    issues=numerical_issues(kind,row)
                    if issues: failures.append({'stable_id':row['stable_id'],'invalid_fields':issues})
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    stream.flush()
                    counts[kind]+=1
        receipt['execution_status']='COMPLETED_WITH_NUMERICAL_FAILURES' if failures else 'COMPLETED'
    except BaseException as error:
        receipt.update(execution_status='INCOMPLETE_EXCEPTION',exception_type=type(error).__name__)
        raise
    finally:
        for kind in kinds:
            path=output/(kind+'.jsonl')
            if path.exists():
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
                hashes[path.name]=digest.hexdigest()
        (output/'run_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'profile':profile,'counts':counts,'failures':len(failures)},ensure_ascii=False))
    return 1 if failures else 0


def main():
    parser=argparse.ArgumentParser(description='运行公开E0复现；不会连接远程服务器。')
    parser.add_argument('--profile',choices=['smoke','full'],default='smoke')
    parser.add_argument('--kind',choices=['all','linear','nonlinear','sequential'],default='all')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    args=parser.parse_args()
    return run(args.output,args.profile,['linear','nonlinear','sequential'] if args.kind=='all' else [args.kind],args.shard_index,args.shards)


if __name__=='__main__': raise SystemExit(main())
