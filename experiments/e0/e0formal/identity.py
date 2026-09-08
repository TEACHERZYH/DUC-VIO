"""E0 v1.4 设置与科学实例的稳定身份。"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple


_KINDS = ("linear", "sequential", "nonlinear")
_STABLE_ID_PATTERN = re.compile(
    r"^v1\.4:E0:(?:(?P<regular>[LS]):s(?P<setting>\d+):r(?P<repeat>\d+)"
    r"|N:g(?P<graph>\d+))$"
)


class IdentityError(ValueError):
    """实例身份不属于冻结 E0 v1.4 范围。"""


def _config_mapping(source: Any) -> Mapping[str, Any]:
    candidate = getattr(source, "config", source)
    if not isinstance(candidate, Mapping):
        raise TypeError("config_or_contract 必须提供配置对象。")
    return candidate


def _json_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise IdentityError(f"{label} 必须是零基非负 JSON 整数。")
    return value


@dataclass(frozen=True)
class FormalInstanceID:
    """一个可独立重算的科学实例，不包含随机流名称。"""

    kind: str
    setting_id: int
    instance_id: int

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise IdentityError(f"未知实例类型：{self.kind!r}")
        _json_nonnegative_int(self.setting_id, "setting_id")
        _json_nonnegative_int(self.instance_id, "instance_id")
        if self.kind == "nonlinear" and self.setting_id != 0:
            raise IdentityError("nonlinear 的冻结 setting_id 必须为 0。")

    @property
    def stable_id(self) -> str:
        """与 dispatcher.TaskSpec 使用相同的稳定字符串。"""

        if self.kind == "nonlinear":
            return f"v1.4:E0:N:g{self.instance_id:03d}"
        prefix = "L" if self.kind == "linear" else "S"
        return (
            f"v1.4:E0:{prefix}:s{self.setting_id:02d}:"
            f"r{self.instance_id:03d}"
        )

    @property
    def logical_id(self) -> str:
        return self.stable_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "setting_id": self.setting_id,
            "instance_id": self.instance_id,
            "stable_id": self.stable_id,
        }

    @classmethod
    def parse(cls, value: str) -> "FormalInstanceID":
        if not isinstance(value, str):
            raise IdentityError("稳定 ID 必须是字符串。")
        match = _STABLE_ID_PATTERN.fullmatch(value)
        if match is None:
            raise IdentityError(f"稳定 ID 格式错误：{value!r}")
        if match.group("graph") is not None:
            identity = cls("nonlinear", 0, int(match.group("graph")))
        else:
            kind = "linear" if match.group("regular") == "L" else "sequential"
            identity = cls(
                kind,
                int(match.group("setting")),
                int(match.group("repeat")),
            )
        # 拒绝非规范的多余前导零，而不是静默归一化不同字符串。
        if identity.stable_id != value:
            raise IdentityError(f"稳定 ID 不是规范形式：{value!r}")
        return identity


@dataclass(frozen=True)
class SettingDefinition:
    """按冻结笛卡尔顺序编号的一项设置。"""

    branch: str
    setting_id: int
    parameters: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"setting_id": self.setting_id}
        result.update(self.parameters)
        return result


def _formal_section(config_or_contract: Any, branch: str) -> Mapping[str, Any]:
    config = _config_mapping(config_or_contract)
    try:
        section = config["formal_contract"][branch]
    except (KeyError, TypeError) as error:
        raise IdentityError(f"配置缺少 formal_contract.{branch}。") from error
    if not isinstance(section, Mapping):
        raise IdentityError(f"formal_contract.{branch} 必须是对象。")
    return section


def _enumerate_settings(
    config_or_contract: Any,
    branch: str,
    expected_axes: Tuple[str, ...],
) -> List[SettingDefinition]:
    section = _formal_section(config_or_contract, branch)
    enumeration = section.get("setting_enumeration")
    if not isinstance(enumeration, Mapping):
        raise IdentityError(f"{branch}.setting_enumeration 缺失。")
    if enumeration.get("index_base") != 0:
        raise IdentityError(f"{branch} 设置必须从 0 开始编号。")
    if tuple(enumeration.get("cartesian_axis_order_outer_to_inner", ())) != expected_axes:
        raise IdentityError(f"{branch} 设置轴顺序与冻结合同不一致。")
    if enumeration.get("last_axis_varies_fastest") is not True:
        raise IdentityError(f"{branch} 最内轴必须变化最快。")
    axis_values: List[Tuple[Any, ...]] = []
    for axis in expected_axes:
        values = section.get(axis)
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, Sequence)
            or not values
        ):
            raise IdentityError(f"{branch}.{axis} 必须是非空数组。")
        axis_values.append(tuple(values))
    settings = [
        SettingDefinition(branch, setting_id, dict(zip(expected_axes, values)))
        for setting_id, values in enumerate(itertools.product(*axis_values))
    ]
    expected_count = section.get("expected_settings")
    if type(expected_count) is not int or len(settings) != expected_count:
        raise IdentityError(f"{branch} 设置数与 expected_settings 不一致。")
    return settings


def enumerate_linear_settings(config_or_contract: Any) -> List[SettingDefinition]:
    return _enumerate_settings(
        config_or_contract,
        "linear",
        ("block_dimensions", "true_noise_multipliers", "mean_scalar_leverages"),
    )


def enumerate_sequential_settings(config_or_contract: Any) -> List[SettingDefinition]:
    return _enumerate_settings(
        config_or_contract,
        "sequential",
        ("block_dimensions", "step_multipliers", "contamination_rates"),
    )


def validate_instance(
    identity: FormalInstanceID, config_or_contract: Any
) -> FormalInstanceID:
    if not isinstance(identity, FormalInstanceID):
        raise TypeError("identity 必须是 FormalInstanceID。")
    section = _formal_section(config_or_contract, identity.kind)
    if identity.kind == "linear":
        settings = section.get("expected_settings")
        instances = section.get("repeats_per_setting")
    elif identity.kind == "sequential":
        settings = section.get("expected_settings")
        instances = section.get("repeats_per_setting")
    else:
        settings = 1
        instances = section.get("graphs")
    if type(settings) is not int or type(instances) is not int:
        raise IdentityError("冻结配置中的实例规模不是整数。")
    if identity.setting_id >= settings or identity.instance_id >= instances:
        raise IdentityError(f"实例超出冻结范围：{identity.stable_id}")
    return identity


def iter_instance_ids(
    config_or_contract: Any, kind: str
) -> Iterator[FormalInstanceID]:
    if kind not in _KINDS:
        raise IdentityError(f"未知实例类型：{kind!r}")
    section = _formal_section(config_or_contract, kind)
    if kind == "nonlinear":
        for instance_id in range(int(section["graphs"])):
            yield FormalInstanceID(kind, 0, instance_id)
        return
    for setting_id in range(int(section["expected_settings"])):
        for instance_id in range(int(section["repeats_per_setting"])):
            yield FormalInstanceID(kind, setting_id, instance_id)


__all__ = [
    "FormalInstanceID",
    "IdentityError",
    "SettingDefinition",
    "enumerate_linear_settings",
    "enumerate_sequential_settings",
    "iter_instance_ids",
    "validate_instance",
]
