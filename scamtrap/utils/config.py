"""YAML config loading with dot-access."""

import yaml
from pathlib import Path


class Config(dict):
    """Dict subclass with attribute access for nested config."""

    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, Config):
                val = Config(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, val):
        self[key] = val


def load_config(path: str) -> Config:
    """Load YAML config file and return as Config object."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(raw)
