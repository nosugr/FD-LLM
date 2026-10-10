"""Read a CWRU configuration section without importing model dependencies."""

from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "cwru.yaml"
SECTIONS = ("preprocessing", "descriptions", "visualization")


def load_section(path: Path, section: str) -> dict:
    """Load unified YAML; also accept legacy flat JSON/YAML configurations."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("CWRU configuration must be a mapping")
    # 合并配置只读取对应分组；没有分组名时按旧的扁平配置解析。
    if any(key in document for key in SECTIONS):
        config = document.get(section)
    else:
        config = document
    if not isinstance(config, dict) or not config:
        raise ValueError(f"CWRU configuration requires a nonempty {section} mapping")
    return config
