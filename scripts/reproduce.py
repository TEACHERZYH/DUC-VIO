"""从论文归档结果重建统计和图表；不进行数据下载或模型训练。"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.public_module.config import load_config
from experiments.public_module.aggregate_public_module import aggregate
from experiments.public_module.aggregate_public_module_v1_6 import aggregate_narrowed


def verify_sources():
    manifest=json.loads((ROOT/'evidence/source_manifest.json').read_text(encoding='utf-8'))
    if not isinstance(manifest.get('files'),list) or len(manifest['files'])<39:
        raise ValueError('来源清单缺少继承材料。')
    seen=set()
    for item in manifest['files']:
        path=Path(item['path'])
        file=(ROOT/path).resolve()
        if path.is_absolute() or '..' in path.parts or not file.is_relative_to(ROOT.resolve()):
            raise ValueError('来源路径超出仓库。')
        if file in seen: raise ValueError('来源清单含重复路径。')
        seen.add(file)
        digest=hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
        if digest.hexdigest()!=item['sha256']:
            raise ValueError('来源校验失败：'+item['path'])
    return len(manifest['files'])


def equal(a,b):
    if isinstance(a,bool) or isinstance(b,bool):
        return type(a) is type(b) and a==b
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def plots(output,gate,summary):
    import matplotlib
    matplotlib.use('Agg')
    from experiments.public_module.make_publication_figure_v1_6 import make_figure
    destination=output/'figures'; destination.mkdir()
    make_figure(gate,summary,destination/'figure2_euroc')
    path=ROOT/'manuscript/figures/formal_e0_v1_4_3/plot_formal_e0_v1_4_3.py'
    spec=importlib.util.spec_from_file_location('e0_archived_plots',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.validate_sources()
    def save(fig,stem):
        import matplotlib.pyplot as plt
        if 'numerical' in stem:
            label='figure3_controlled'; fig.set_size_inches(7.2,2.8)
            fig._suptitle.set_text('Controlled numerical evaluation')
            fig.axes[0].set_title('Probe count selection',loc='left',fontweight='bold',pad=7)
            fig.axes[1].legend(loc='lower right',fontsize=5.8,labelspacing=.1,
                               handletextpad=.4,borderpad=0,borderaxespad=.1)
            for item in fig.axes[2].texts:
                if item.get_text().startswith('-') and item.get_text().endswith('%'):
                    x,y=item.get_position(); item.set_position(((x+1.2+100)/2,y)); item.set_ha('center')
        else:
            label='figure4_sequential'; fig._suptitle.set_text('Sequential scale tracking')
            for ax in fig.axes:
                if 'Frozen setting ID' in ax.get_xlabel(): ax.set_xlabel('Setting ID')
        for ext in ['svg','pdf','png']:
            fig.savefig(destination/(label+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.045)
        plt.close(fig)
        return {}
    mod.save_all=save; mod.figure1(); mod.figure2()


def reproduce(output,make_plots=False):
    count=verify_sources()
    output.mkdir(parents=True,exist_ok=False)
    cfg=ROOT/'experiments/contracts/public_module_config_v1.5.json'
    scope=ROOT/'experiments/contracts/public_module_scope_v1.6.json'
    source=ROOT/'experiments/artifacts/public_module_formal_v1_5'
    six=output/'euroc_six_sequences'; five=output/'euroc_five_sequences'
    six_return=aggregate(load_config(cfg),source,six)
    five_return=aggregate_narrowed(cfg,scope,source,five)
    keys=['status','checks','sequence_effects','equal_sequence_median_differences',
          'descriptive_cluster_bootstrap_95_percentile_intervals']
    matches={}
    for version,folder in [('v1_5',six),('v1_6',five)]:
        original=ROOT/('experiments/artifacts/public_module_aggregate_'+version)
        expected=json.loads((original/'gate_review.json').read_text(encoding='utf-8'))
        obtained=json.loads((folder/'gate_review.json').read_text(encoding='utf-8'))
        matches[version]=all(equal(obtained[k],expected[k]) for k in keys)
        for name in ['sequence_method_summary.csv','failure_counts.csv']:
            with (original/name).open(encoding='utf-8',newline='') as f: ref=list(csv.DictReader(f))
            with (folder/name).open(encoding='utf-8',newline='') as f: cur=list(csv.DictReader(f))
            def normalize(rows):
                converted=[]
                for row in rows:
                    item={}
                    for key,value in row.items():
                        try: item[key]=float(value) if value else value
                        except ValueError: item[key]=value
                    converted.append(item)
                return converted
            matches[version+'_'+name]=equal(normalize(cur),normalize(ref))
    # 进程退出码表示汇总程序是否正常完成；科学门槛状态由 gate_review.json 核验。
    if not all(matches.values()) or five_return!=0 or six_return!=0:
        raise ValueError('重算结果不匹配归档的五序列通过 / 六序列未通过状态：'+str(matches))
    table_dir=output/'tables'; table_dir.mkdir()
    from experiments.public_module.make_publication_tables_v1_6 import make_tables
    make_tables(five/'gate_review.json',five/'sequence_method_summary.csv',table_dir/'euroc')
    e0_table_dir=table_dir/'e0'; e0_table_dir.mkdir()
    for source in (ROOT/'manuscript/source_data/formal_e0_v1_4_2').glob('*.csv'):
        shutil.copyfile(source,e0_table_dir/source.name)
    if make_plots: plots(output,five/'gate_review.json',five/'sequence_method_summary.csv')
    report={'status':'PASS','verified_source_files':count,'matches':matches,'new_scientific_runs':0,
            'euroc':'Recomputed from all six sequences of archived per-pair metrics; both fit sizes retained.',
            'e0':'Verified archived summaries and plotting source tables; full atomic E0 packs are not included.',
            'plots_created':make_plots}
    (output/'reproduction_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','verified_sources':count,'matched_checks':len(matches),'plots':make_plots}))


def main():
    p=argparse.ArgumentParser(description='离线重算论文结果；输出目录必须不存在。')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--plots',action='store_true')
    args=p.parse_args(); reproduce(args.output,args.plots)


if __name__=='__main__': main()
