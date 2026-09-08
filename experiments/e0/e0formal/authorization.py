"""公开离线运行会话；与原服务器授权系统没有等价关系。"""
from dataclasses import dataclass
from .contract import FrozenContract, FormalExecutionLocked
from .identity import FormalInstanceID, validate_instance


@dataclass(frozen=True)
class FormalProcessSession:
    contract: FrozenContract

    def verify(self,config,stable_id):
        if config is not self.contract:
            raise FormalExecutionLocked('运行会话和配置不匹配。')
        config.verify()
        validate_instance(FormalInstanceID.parse(stable_id),config)


class BatchAuthorization:
    def __init__(self,*args,**kwargs):
        raise FormalExecutionLocked('本公开包不提供原服务器的批次授权或计时预热接口。请使用公开运行入口。')
