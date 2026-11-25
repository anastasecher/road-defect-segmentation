from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_KB = Path(__file__).resolve().parents[1] / "configs" / "knowledge_base.json"

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
       "==": operator.eq, "!=": operator.ne}

POTHOLE, CRACK, MANHOLE = 1, 2, 3


def extract_facts(mask: np.ndarray, conf: np.ndarray | None = None, near_zone_from: float = 0.6) -> dict:
    h, w = mask.shape
    total = float(h * w)
    facts: dict = {}
    for cid, name in ((POTHOLE, "pothole"), (CRACK, "crack"), (MANHOLE, "manhole")):
        n, lab, st, cen = cv2.connectedComponentsWithStats((mask == cid).astype(np.uint8))
        areas = st[1:, cv2.CC_STAT_AREA].astype(float) if n > 1 else np.zeros(0)
        facts[f"{name}_count"] = int(n - 1)
        facts[f"{name}_area_pct"] = round(100 * areas.sum() / total, 3)
        facts[f"{name}_max_pct"] = round(100 * areas.max() / total, 3) if len(areas) else 0.0
        if name == "pothole":
            near = [a for a, (_, cy) in zip(areas, cen[1:]) if cy >= near_zone_from * h]
            facts["pothole_near_max_pct"] = round(100 * max(near) / total, 3) if near else 0.0
    facts["defect_area_pct"] = round(facts["pothole_area_pct"] + facts["crack_area_pct"], 3)
    defect = (mask == POTHOLE) | (mask == CRACK)
    facts["mean_conf"] = round(float(conf[defect].mean()), 3) if conf is not None and defect.any() else 1.0
    return facts


def _fmt(value, raw=None) -> str:
    text = {True: "да", False: "нет", None: "нет данных"}.get(value, value) if isinstance(value, (bool, type(None))) \
        else f"{value:g}" if isinstance(value, float) else str(value)
    if isinstance(raw, str) and raw.startswith("$"):
        text += f" [{raw[1:]}]"
    return text


@dataclass
class InferenceResult:
    level: int
    level_name: str
    priority: str
    messages: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    fired: list[dict] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def explanation(self) -> str:
        lines = []
        for f in self.fired:
            conds = " И ".join(f"{fact} {op} {_fmt(val, raw)} (факт: {_fmt(cur)})"
                               for fact, op, raw, cur, val in f["conditions"])
            lines.append(f"{f['id']} «{f['name']}»: ЕСЛИ {conds}")
        return "\n".join(lines)


class KnowledgeBase:
    def __init__(self, path: str | Path = DEFAULT_KB):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.params: dict = data["params"]
        self.param_descriptions: dict = data.get("param_descriptions", {})
        self.rules: list[dict] = data["rules"]
        self.levels: dict = {int(k): v for k, v in data["severity_levels"].items()}
        self._validate()

    def _validate(self):
        ids = [r["id"] for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Повторяющиеся id правил в базе знаний")
        for r in self.rules:
            for fact, op, val in r["if"]:
                if op not in OPS:
                    raise ValueError(f"{r['id']}: неизвестная операция {op}")
                if isinstance(val, str) and val.startswith("$") and val[1:] not in self.params:
                    raise ValueError(f"{r['id']}: нет параметра {val}")

    def resolve(self, value):
        if isinstance(value, str) and value.startswith("$"):
            return self.params[value[1:]]
        return value


class InferenceEngine:

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def _check(self, rule: dict, facts: dict):
        used = []
        for fact, op, raw in rule["if"]:
            value = self.kb.resolve(raw)
            current = facts.get(fact)
            if current is None:
                if op == "!=":
                    used.append((fact, op, raw, None, value))
                    continue
                return None
            if not OPS[op](current, value):
                return None
            used.append((fact, op, raw, current, value))
        return used

    def run(self, facts: dict, max_iter: int = 20) -> InferenceResult:
        facts = dict(facts)
        fired_ids: set[str] = set()
        fired, levels, messages, recs = [], [], [], []
        for _ in range(max_iter):
            new_fire = False
            for rule in self.kb.rules:
                if rule["id"] in fired_ids:
                    continue
                used = self._check(rule, facts)
                if used is None:
                    continue
                fired_ids.add(rule["id"])
                new_fire = True
                then = rule["then"]
                facts.update(then.get("facts", {}))
                if "severity" in then:
                    levels.append(then["severity"])
                if then.get("message"):
                    messages.append(then["message"])
                if then.get("recommendation") and then["recommendation"] not in recs:
                    recs.append(then["recommendation"])
                fired.append({"id": rule["id"], "name": rule["name"], "conditions": used,
                              "severity": then.get("severity")})
            if not new_fire:
                break
        level = max(levels) if levels else 0
        info = self.kb.levels[level]
        return InferenceResult(level, info["name"], info["priority"], messages, recs, fired, facts)


def assess(mask: np.ndarray, conf: np.ndarray | None = None, kb: KnowledgeBase | None = None) -> InferenceResult:
    kb = kb or KnowledgeBase()
    facts = extract_facts(mask, conf, kb.params.get("near_zone_from", 0.6))
    return InferenceEngine(kb).run(facts)
