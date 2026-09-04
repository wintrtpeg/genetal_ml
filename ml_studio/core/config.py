"""설정 외부화.

화면에서 고른 값을 파일 하나로 떨어뜨리고, 그 파일만으로 같은 실행을 다시 만든다.
Dataiku Scenario 처럼 UI 가 없는 곳에서 돌릴 때 필요하고, 사람 사이에 설정을
주고받을 때도 필요하다.

원칙
1. dataclass 를 진실의 원천으로 둔다. YAML 은 그 표현일 뿐이다.
2. 모르는 키는 조용히 버리지 않고 경고로 돌려준다. 오타 하나가 조용히 무시되면
   "설정을 바꿨는데 결과가 같다"가 된다.
3. YAML 이 없으면 JSON 으로 떨어진다. 폐쇄망에서 pyyaml 이 없을 수 있다.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from .features import FeatureConfig
from .preprocess import PreprocessConfig
from .train import TrainConfig
from .validation import SplitConfig

__all__ = [
    "StudioConfig", "to_dict", "from_dict", "dump", "load",
    "dumps", "loads", "diff", "SECTIONS",
]

SECTIONS = {
    "features": FeatureConfig,
    "preprocess": PreprocessConfig,
    "split": SplitConfig,
    "train": TrainConfig,
}

SCHEMA_VERSION = 1


def _has_yaml() -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


def _plain(v: Any) -> Any:
    """dataclass·tuple·Path 를 YAML/JSON 이 받는 형태로 낮춘다."""
    import pandas as pd

    if is_dataclass(v) and not isinstance(v, type):
        return {f.name: _plain(getattr(v, f.name)) for f in fields(v)}
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if hasattr(v, "item") and getattr(v, "size", 1) == 1:
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v


class StudioConfig:
    """네 개의 설정 dataclass 를 한 묶음으로 다룬다."""

    def __init__(
        self,
        features: FeatureConfig | None = None,
        preprocess: PreprocessConfig | None = None,
        split: SplitConfig | None = None,
        train: TrainConfig | None = None,
        meta: dict | None = None,
    ):
        self.features = features or FeatureConfig()
        self.preprocess = preprocess or PreprocessConfig()
        self.split = split or SplitConfig()
        self.train = train or TrainConfig()
        self.meta = meta or {}

    def __eq__(self, other) -> bool:
        if not isinstance(other, StudioConfig):
            return NotImplemented
        return to_dict(self) == to_dict(other)

    def __repr__(self) -> str:
        return (f"StudioConfig(target={self.meta.get('target')!r}, "
                f"unseen_ratio={self.split.unseen_ratio}, "
                f"fold_selection={self.train.fold_selection})")


def to_dict(cfg: StudioConfig) -> dict:
    """직렬화 가능한 사전으로 낮춘다. train.split 은 split 과 중복이라 뺀다."""
    out: dict = {"schema_version": SCHEMA_VERSION, "meta": _plain(cfg.meta)}
    for name in SECTIONS:
        out[name] = _plain(getattr(cfg, name))
    out["train"].pop("split", None)
    return out


def _build(cls, data: dict, section: str, warnings: list[str]):
    """dataclass 를 만들되 모르는 키는 경고로 남긴다."""
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in (data or {}).items():
        if k not in known:
            warnings.append(f"{section}.{k} — 모르는 설정입니다 (무시됨)")
            continue
        kwargs[k] = v
    obj = cls()
    for k, v in kwargs.items():
        cur = getattr(obj, k)
        if isinstance(cur, tuple) and isinstance(v, list):
            v = tuple(v)
        obj = replace(obj, **{k: v})
    return obj


def from_dict(data: dict) -> tuple[StudioConfig, list[str]]:
    """사전에서 설정을 복원한다. (설정, 경고 목록) 반환."""
    warnings: list[str] = []
    ver = data.get("schema_version")
    if ver is not None and ver != SCHEMA_VERSION:
        warnings.append(
            f"schema_version {ver} — 이 버전은 {SCHEMA_VERSION} 을 씁니다. "
            "달라진 항목은 기본값으로 채워집니다.")

    built = {name: _build(cls, data.get(name, {}), name, warnings)
             for name, cls in SECTIONS.items()}
    # TrainConfig.split 은 파일에 없다. split 섹션을 넣어 하나로 유지한다.
    built["train"] = replace(built["train"], split=built["split"])

    unknown = set(data) - set(SECTIONS) - {"schema_version", "meta"}
    for k in sorted(unknown):
        warnings.append(f"{k} — 모르는 최상위 항목입니다 (무시됨)")

    return StudioConfig(meta=data.get("meta", {}), **built), warnings


def dumps(cfg: StudioConfig, prefer_yaml: bool = True) -> str:
    """문자열로 직렬화한다. pyyaml 이 없으면 JSON."""
    data = to_dict(cfg)
    if prefer_yaml and _has_yaml():
        import yaml
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False,
                              default_flow_style=False)
    return json.dumps(data, ensure_ascii=False, indent=2)


def loads(text: str) -> tuple[StudioConfig, list[str]]:
    """문자열에서 복원한다. YAML 과 JSON 을 모두 받는다."""
    text = text.strip()
    if not text:
        raise ValueError("빈 설정입니다.")
    if _has_yaml():
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("설정 파일의 최상위는 사전이어야 합니다.")
    return from_dict(data)


def dump(cfg: StudioConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps(cfg, prefer_yaml=p.suffix in (".yaml", ".yml")), encoding="utf-8")
    return p


def load(path: str | Path) -> tuple[StudioConfig, list[str]]:
    return loads(Path(path).read_text(encoding="utf-8"))


def diff(a: StudioConfig, b: StudioConfig):
    """두 설정의 차이만 표로 낸다. 무엇을 바꿨는지 되짚을 때 쓴다."""
    import pandas as pd

    da, db = to_dict(a), to_dict(b)
    rows = []
    for section in SECTIONS:
        sa, sb = da.get(section, {}), db.get(section, {})
        for key in sorted(set(sa) | set(sb)):
            va, vb = sa.get(key), sb.get(key)
            if va != vb:
                rows.append({"섹션": section, "항목": key, "A": va, "B": vb})
    return pd.DataFrame(rows)
