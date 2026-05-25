from dataclasses import dataclass, fields, replace
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


@dataclass(frozen=True)
class IVIDConfig:
    dataset_name: str = "mosi"
    seed: int = 1
    batch_size: int = 8
    learning_rate: float = 5e-6
    epochs: int = 30
    patience: int = 8
    dropout: float = 0.3
    text_context_len: int = 2
    audio_context_len: int = 1
    hidden_size: int = 1024
    video_feature_dim: int = 20
    num_attention_heads: int = 16
    temperature: float = 0.07
    pure_weight: float = 0.1
    bias_weight: float = 0.1
    complete_weight: float = 0.1
    log: str = "logs/ivid.log"
    checkpoint_dir: str = "checkpoint"
    save_model: bool = True


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = _load_yaml(file)

    valid_fields = {field.name for field in fields(IVIDConfig)}
    unknown_fields = sorted(set(data) - valid_fields)
    if unknown_fields:
        names = ", ".join(unknown_fields)
        raise ValueError(f"Unknown config field(s): {names}")

    return IVIDConfig(**data)


def update_config(config, **overrides):
    valid_fields = {field.name for field in fields(config)}
    cleaned = {
        key: value
        for key, value in overrides.items()
        if key in valid_fields and value is not None
    }
    return replace(config, **cleaned)


def _load_yaml(file):
    if yaml is not None:
        return yaml.safe_load(file) or {}

    data = {}
    for line in file:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
