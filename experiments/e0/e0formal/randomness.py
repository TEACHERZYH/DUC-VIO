"""E0 v1.4 的独立随机子流与可复核种子派生。"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any, Dict, Mapping, Sequence

from .identity import FormalInstanceID, validate_instance


class RandomnessContractError(ValueError):
    """随机流请求不符合冻结合同。"""


class SeedVectorMismatch(RandomnessContractError):
    """实现未通过冻结的种子测试向量。"""


def _config_mapping(source: Any) -> Mapping[str, Any]:
    candidate = getattr(source, "config", source)
    if not isinstance(candidate, Mapping):
        raise TypeError("config_or_contract 必须提供配置对象。")
    return candidate


def _normalize_json(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RandomnessContractError("规范 JSON 不允许 NaN 或无穷值。")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise RandomnessContractError(
        f"规范 JSON 数组含不支持的类型：{type(value).__name__}"
    )


def canonical_json_text(value: Sequence[Any]) -> str:
    """按冻结规则生成 NFC、紧凑、非 ASCII 转义关闭的 JSON 数组。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RandomnessContractError("种子载荷必须是 JSON 数组。")
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Sequence[Any]) -> bytes:
    return canonical_json_text(value).encode("utf-8")


def _randomness_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        section = config["formal_contract"]["randomness"]
    except (KeyError, TypeError) as error:
        raise RandomnessContractError(
            "配置缺少 formal_contract.randomness。"
        ) from error
    if not isinstance(section, Mapping):
        raise RandomnessContractError("randomness 必须是对象。")
    return section


def seed_payload(
    config_or_contract: Any,
    component: str,
    identity: FormalInstanceID,
    subcomponent_id: str = "none",
) -> list:
    """构造顺序固定的 9 元素种子身份载荷。"""

    config = _config_mapping(config_or_contract)
    validate_instance(identity, config)
    randomness = _randomness_section(config)
    streams = randomness.get("named_component_streams")
    if (
        isinstance(streams, (str, bytes))
        or not isinstance(streams, Sequence)
        or component not in streams
    ):
        raise RandomnessContractError(f"未冻结的随机流名称：{component!r}")
    if not isinstance(subcomponent_id, str):
        raise RandomnessContractError("subcomponent_id 必须是 JSON 字符串。")
    derived = randomness.get("derived_seed")
    if not isinstance(derived, Mapping):
        raise RandomnessContractError("配置缺少 derived_seed。")
    payload_rule = derived.get("payload")
    if not isinstance(payload_rule, Mapping):
        raise RandomnessContractError("配置缺少 derived_seed.payload。")
    expected_serialization = {
        "format": "canonical_json_array",
        "encoding": "UTF-8",
        "unicode_normalization": "NFC",
        "ensure_ascii": False,
        "allow_nan": False,
        "compact_separators": True,
    }
    if any(payload_rule.get(key) != expected for key, expected in expected_serialization.items()):
        raise RandomnessContractError("种子载荷序列化规则与冻结合同不一致。")
    prefix = payload_rule.get("namespace_prefix")
    if list(prefix) != ["DUC-VIO", "E0", "v1.4", 91027]:
        raise RandomnessContractError("namespace_prefix 与冻结合同不一致。")
    if list(payload_rule.get("identity_suffix_order", ())) != [
        "component",
        "instance_kind",
        "setting_id",
        "instance_id",
        "subcomponent_id",
    ]:
        raise RandomnessContractError("种子身份字段顺序与冻结合同不一致。")
    if (
        payload_rule.get("missing_subcomponent_representation")
        != "none_string_sentinel"
    ):
        raise RandomnessContractError("缺省子组件标记与冻结合同不一致。")
    return list(prefix) + [
        component,
        identity.kind,
        identity.setting_id,
        identity.instance_id,
        subcomponent_id,
    ]


def seed_receipt(
    config_or_contract: Any,
    component: str,
    identity: FormalInstanceID,
    subcomponent_id: str = "none",
) -> Dict[str, Any]:
    """返回可审计载荷、规范串、完整摘要和 uint64 种子。"""

    config = _config_mapping(config_or_contract)
    randomness = _randomness_section(config)
    derived = randomness.get("derived_seed")
    if not isinstance(derived, Mapping):
        raise RandomnessContractError("配置缺少 derived_seed。")
    if (
        randomness.get("bit_generator") != "PCG64"
        or derived.get("hash_algorithm") != "SHA-256"
        or derived.get("digest_prefix_bytes") != 8
        or derived.get("byte_order") != "big"
        or derived.get("signed") is not False
    ):
        raise RandomnessContractError("派生种子规则与冻结合同不一致。")
    payload = seed_payload(config, component, identity, subcomponent_id)
    canonical = canonical_json_text(payload)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return {
        "payload": payload,
        "canonical_json": canonical,
        "sha256": digest.hex(),
        "derived_seed_uint64": seed,
    }


def derive_seed(
    config_or_contract: Any,
    component: str,
    identity: FormalInstanceID,
    subcomponent_id: str = "none",
) -> int:
    return int(
        seed_receipt(
            config_or_contract, component, identity, subcomponent_id
        )["derived_seed_uint64"]
    )


def rng_for(
    config_or_contract: Any,
    component: str,
    identity: FormalInstanceID,
    subcomponent_id: str = "none",
) -> Any:
    """建立只属于该身份的 NumPy PCG64 Generator；NumPy 为延迟依赖。"""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - 正式环境应预装 NumPy
        raise RandomnessContractError("rng_for 需要 NumPy。") from error
    seed = derive_seed(config_or_contract, component, identity, subcomponent_id)
    return np.random.Generator(np.random.PCG64(seed))


def verify_seed_test_vectors(config_or_contract: Any) -> tuple:
    """核对三组冻结向量；成功时返回三个 uint64。"""

    config = _config_mapping(config_or_contract)
    randomness = _randomness_section(config)
    try:
        vectors = randomness["derived_seed"]["test_vectors"]
    except (KeyError, TypeError) as error:
        raise SeedVectorMismatch("配置缺少种子测试向量。") from error
    if (
        isinstance(vectors, (str, bytes))
        or not isinstance(vectors, Sequence)
        or len(vectors) != 3
    ):
        raise SeedVectorMismatch("冻结合同必须提供恰好三组种子测试向量。")
    observed_seeds = []
    for index, vector in enumerate(vectors):
        if not isinstance(vector, Mapping) or set(vector) != {
            "payload", "derived_seed_uint64"
        }:
            raise SeedVectorMismatch(f"测试向量 {index} 结构错误。")
        payload = vector["payload"]
        if (
            isinstance(payload, (str, bytes))
            or not isinstance(payload, Sequence)
            or len(payload) != 9
        ):
            raise SeedVectorMismatch(f"测试向量 {index} 载荷错误。")
        identity = FormalInstanceID(payload[5], payload[6], payload[7])
        reconstructed = seed_payload(config, payload[4], identity, payload[8])
        if reconstructed != list(payload):
            raise SeedVectorMismatch(f"测试向量 {index} 身份模板不一致。")
        observed = derive_seed(config, payload[4], identity, payload[8])
        if observed != vector["derived_seed_uint64"]:
            raise SeedVectorMismatch(
                f"测试向量 {index} 不匹配：期望 "
                f"{vector['derived_seed_uint64']}，实际 {observed}"
            )
        observed_seeds.append(observed)
    return tuple(observed_seeds)


__all__ = [
    "RandomnessContractError",
    "SeedVectorMismatch",
    "canonical_json_bytes",
    "canonical_json_text",
    "derive_seed",
    "rng_for",
    "seed_payload",
    "seed_receipt",
    "verify_seed_test_vectors",
]
