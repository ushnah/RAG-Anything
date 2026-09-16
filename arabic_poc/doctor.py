"""Check local prerequisites without downloading models or printing secrets."""
import importlib.util
import json
import os
import shutil
from pathlib import Path


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    modules = ['raganything', 'lightrag', 'FlagEmbedding', 'openai', 'whisper',
               'paddleocr', 'paddle', 'pypdfium2', 'torch', 'transformers']
    print(json.dumps({
        'python_modules': {name: importlib.util.find_spec(name) is not None for name in modules},
        'executables': {name: bool(shutil.which(name)) for name in ['ffmpeg', 'ffprobe']},
        'generation_configured': bool(os.getenv('QWEN_BASE_URL') and os.getenv('QWEN_MODEL')),
        'local_qwen_vl_config': Path('models/qwen2.5-vl-3b/config.json').exists(),
        'note': 'Presence checks only; does not verify model weights, imports, endpoint health, or credentials.'
    }, indent=2))


if __name__ == '__main__':
    main()
