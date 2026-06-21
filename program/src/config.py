from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Tiny YAML subset parser used only when PyYAML is unavailable.

    It supports the repository's indentation-based mappings and bracket lists.
    PyYAML remains the preferred parser whenever installed.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        content = raw_line.split(" #", 1)[0].rstrip()
        indent = len(content) - len(content.lstrip(" "))
        if ":" not in content:
            continue
        key, value = content.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _yaml_load(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return _simple_yaml_load(text)


def _simple_yaml_dump(data: Any, indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(_simple_yaml_dump(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(value)}")
    else:
        lines.append(f"{prefix}{_format_scalar(data)}")
    return "\n".join(line for line in lines if line != "")


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    if isinstance(value, str):
        if value == "" or any(ch in value for ch in [":", "#", "{", "}", "[", "]"]):
            return json.dumps(value)
        return value
    return str(value)


def yaml_dump(data: dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(data, sort_keys=False)
    except Exception:
        return _simple_yaml_dump(data) + "\n"


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    data = _yaml_load(path.read_text(encoding="utf-8"))
    data["_config_path"] = str(path)
    data["_project_root"] = str(path.resolve().parent.parent)
    return data


def write_config(config: dict[str, Any], path: str | Path) -> None:
    sanitized = {k: v for k, v in config.items() if not k.startswith("_")}
    Path(path).write_text(yaml_dump(sanitized), encoding="utf-8")


def resolve_project_path(config: dict[str, Any], maybe_path: str | None) -> Path | None:
    if maybe_path in {None, ""}:
        return None
    path = Path(str(maybe_path))
    if path.is_absolute():
        return path
    root = Path(config.get("_project_root", "."))
    return root / path


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def get_dotted(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def load_experiment_matrix(matrix_path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    matrix = load_config(matrix_path)
    matrix_root = Path(matrix_path).resolve().parent
    base_path = matrix.get("base_config", "configs/default.yaml")
    base_config_path = Path(base_path)
    if not base_config_path.is_absolute():
        base_config_path = (matrix_root / ".." / base_path).resolve()
        if not base_config_path.exists():
            base_config_path = (matrix_root / base_path).resolve()
    base = load_config(base_config_path)
    scenarios: dict[str, dict[str, Any]] = {}
    for name, override in (matrix.get("scenarios") or {}).items():
        scenarios[name] = deep_update(base, override or {})
    grid = matrix.get("parameter_grid") or {}
    if grid:
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            scenario = copy.deepcopy(base)
            label_parts = []
            for key, value in zip(keys, values, strict=True):
                set_dotted(scenario, key, value)
                label_parts.append(f"{key.split('.')[-1]}={value}")
            scenarios["sweep_" + "_".join(label_parts).replace(".", "p")] = scenario
    return base, scenarios

