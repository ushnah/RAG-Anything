"""Load application prompts independently of the process working directory."""
from functools import lru_cache
from pathlib import Path
from string import Template
import yaml


@lru_cache(maxsize=1)
def _prompts():
    path = Path(__file__).with_name('prompts.yaml')
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                                         for k, v in data.items()):
        raise ValueError(f'Invalid prompt catalog: {path}')
    return data


def get_prompt(key, **values):
    """Render explicit ${name} placeholders; literal JSON braces remain untouched."""
    text = _prompts()[key]
    return Template(text).substitute(values) if values else text
