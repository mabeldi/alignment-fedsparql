"""Pipeline settings fixed for the server lifetime; restart after editing YAML."""

from pathlib import Path

import yaml


_CONFIG_PATH = Path(__file__).with_name("pipeline.yaml")


def _read_alignment_setting(path: Path = _CONFIG_PATH) -> bool:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or type(config.get("alignments_enabled")) is not bool:
        raise ValueError(f"{path}: alignments_enabled must be a YAML boolean (true or false).")
    return config["alignments_enabled"]


_ALIGNMENTS_ENABLED = _read_alignment_setting()


def alignments_enabled() -> bool:
    """Whether every pipeline stage may read the schema mapping dataset."""
    return _ALIGNMENTS_ENABLED
