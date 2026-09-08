"""公开复现用配置适配层；不代表原服务器执行授权。

科学计算模块保持原文件字节。此适配层只验证公开配置和稳定任务身份。
本包不连接、申请或调度任何远程资源。
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any


class FormalExecutionLocked(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenContract:
    config: dict[str, Any]
    config_path: Path
    payload_sha256: str

    def verify(self):
        current=hashlib.sha256(json.dumps(self.config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if current!=self.payload_sha256:
            raise FormalExecutionLocked('加载后的科学配置已变更。')


def load_frozen_contract(config_path: Path | None=None):
    root=Path(__file__).resolve().parents[3]
    path=config_path or root/'experiments/e0/config/e0_v1_4.json'
    manifest=json.loads((root/'evidence/source_manifest.json').read_text(encoding='utf-8'))
    item=next(r for r in manifest['files'] if r['path']=='experiments/e0/config/e0_v1_4.json')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise FormalExecutionLocked('配置不匹配公开冻结版本。')
    data=json.loads(path.read_text(encoding='utf-8'))
    fingerprint=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return FrozenContract(data,path,fingerprint)
