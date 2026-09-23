"""Arbisoft gateway adapters. No local neural model is loaded here."""
import base64
import io
import json
import logging
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
ENTITY_CATEGORIES = ('people', 'landmarks', 'places', 'organizations', 'events')


def empty_entities():
    return {category: [] for category in ENTITY_CATEGORIES}


def normalize_entities(value):
    entities = empty_entities()
    if not isinstance(value, dict):
        return entities
    for category in ENTITY_CATEGORIES:
        values = value.get(category, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        entities[category] = list(dict.fromkeys(
            item.strip() for item in values
            if isinstance(item, str) and item.strip()))
    return entities


def flatten_entities(entities):
    return '\n'.join(name for category in ENTITY_CATEGORIES for name in entities.get(category, []))


def parse_visual_response(raw):
    """Return a description and safe entity lists without rejecting the description."""
    if not isinstance(raw, str):
        logger.warning('Remote vision returned a non-text description response')
        return '', empty_entities()
    cleaned = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw.strip(), flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        logger.warning('Remote vision returned malformed description JSON; preserving raw description')
        return raw.strip(), empty_entities()
    if not isinstance(payload, dict):
        logger.warning('Remote vision description JSON was not an object; preserving raw description')
        return raw.strip(), empty_entities()
    description = payload.get('description')
    if not isinstance(description, str) or not description.strip():
        logger.warning('Remote vision JSON omitted a usable description; preserving raw response')
        description = raw.strip()
    return description.strip(), normalize_entities(payload.get('entities'))


def enabled(role=None):
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    return os.getenv(role.upper() + '_BACKEND' if role else 'MODEL_BACKEND', os.getenv('MODEL_BACKEND','local')) == 'remote'


def request(endpoint, **kwargs):
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    base = os.environ['REMOTE_BASE_URL'].rstrip('/')
    try:
        response = httpx.post(base + endpoint,
            headers={'Authorization': 'Bearer ' + os.environ['REMOTE_API_KEY']},
            timeout=float(os.getenv('REMOTE_TIMEOUT', '180')), **kwargs)
        if not response.is_success:
            # Gateway bodies can echo credentials. Do not expose them.
            raise RuntimeError(f'Remote {endpoint} returned HTTP {response.status_code}')
        return response.json()
    except httpx.RequestError as exc:
        raise RuntimeError(f'Remote connection failed: {type(exc).__name__}') from None


def chat(messages, model=None, json_output=False):
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    payload = {'model': model or os.getenv('REMOTE_TEXT_MODEL', 'cerebras/gpt-oss-120b'),
               'messages': messages, 'temperature': 0, 'max_tokens': int(os.getenv('REMOTE_MAX_TOKENS', '4096'))}
    if json_output:
        payload['response_format'] = {'type': 'json_object'}
    result = request('/chat/completions', json=payload)
    choice = result['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise RuntimeError('Remote generation reached its token limit; response rejected.')
    content = choice['message'].get('content')
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('Remote model returned no answer text.')
    return content


class RemoteVision:
    def __init__(self, describe=False):
        self.describe = describe

    def parse_image(self, path, lang='ar'):
        from PIL import Image
        with Image.open(path) as source:
            picture = source.convert('RGB')
            picture.thumbnail((1280,1280))
            output = io.BytesIO()
            picture.save(output, format='JPEG')
        prompt = ('أجب بالعربية وبصيغة JSON فقط بهذا الشكل: '
              '{"description":"المشهد: ... | العناصر والعدد المرئي: ... | النص المرئي: ... | الأشخاص/المعالم والأسماء: ... | الموضع/الألوان والتفاصيل المعمارية: ...", '
              '"entities":{"people":[],"landmarks":[],"places":[],"organizations":[],"events":[]}}. '
              'حافظ على وصف قصير وقابل للبحث ولا تتجاوز 110 كلمات. اذكر العدد فقط عندما يكون الشيء كاملاً وواضحاً؛ وإلا اكتب «العدد غير محسوم». '
              'اكتب «لا يوجد نص مقروء» إن لم يظهر نص. استخرج أسماء الأشخاص أو المؤسسات أو الأحداث إذا كانت مكتوبة أو مدعومة بسياق موثوق؛ لا تخمن. '
              'يمكن تسمية مبنى أو معلم عندما تكون خصائصه البصرية مميزة بما يكفي، وإلا اتركه خارج القائمة. لا تحدد هوية شخص من وجهه أو مظهره وحده. '
              'احتفظ بالأسماء باللغة أو الخط الظاهر في الصورة، واجعل كل قيمة في القوائم اسماً قصيراً. إذا لم تجد كيانات فأعد قوائم فارغة. '
              'لا تستنتج الحركة أو الموقع من صورة ثابتة.'
              if self.describe else 'انسخ النص المرئي فقط كما هو دون تصحيح أو تلخيص أو إكمال. لا تخمن النص غير المقروء. أعد JSON فقط بالشكل {"text":"النص"}. إذا لم تجد نصاً مقروءاً اجعل text فارغاً، ولا تضع شرحاً.')
        content = [{'type':'text','text':prompt}, {'type':'image_url','image_url':{
            'url':'data:image/jpeg;base64,'+base64.b64encode(output.getvalue()).decode()}}]
        try:
            result = chat([{'role':'user','content':content}], os.environ['REMOTE_VISION_MODEL'], json_output=True)
        except Exception:
            if not self.describe:
                raise
            logger.exception('Remote structured visual description failed; retrying legacy description mode')
            legacy_prompt = ('أنشئ وصفاً قصيراً قابلاً للبحث بالعربية في خمس خانات بهذا الترتيب: '
                             'المشهد: ... | العناصر والعدد المرئي: ... | النص المرئي: ... | الأشخاص/المعالم والأسماء: ... | الموضع/الألوان والتفاصيل المعمارية: ... . '
                             'اذكر العدد فقط عندما يكون واضحاً، ولا تحدد هوية شخص من وجهه أو مظهره. '
                             'سمّ المبنى أو المعلم فقط إن دعمته خصائص بصرية مميزة أو نص واضح؛ وإلا صفه دون تخمين. لا تتجاوز 110 كلمات.')
            content[0]['text'] = legacy_prompt
            result = chat([{'role':'user','content':content}], os.environ['REMOTE_VISION_MODEL'], json_output=False)
        if not self.describe:
            try:
                result = json.loads(result)['text']
                if not isinstance(result, str): raise ValueError()
            except (ValueError,KeyError,TypeError):
                raise RuntimeError('Remote OCR returned invalid structured text.') from None
            return [{'text':result, 'page_idx':0}]
        description, entities = parse_visual_response(result)
        return [{'text':description, 'description':description, 'entities':entities, 'page_idx':0}]

    def parse_pdf(self, path, lang='ar'):
        from .models import QwenParser
        return QwenParser.parse_pdf(self, path, lang)


class RemoteASR:
    def transcribe(self, path, **kwargs):
        with open(path, 'rb') as file:
            result = request('/audio/transcriptions', data={'model':os.getenv('REMOTE_ASR_MODEL','whisper'),
                'language':'ar','response_format':'verbose_json','timestamp_granularities[]':'segment'},
                files={'file':(Path(path).name,file,'application/octet-stream')})
        if not isinstance(result.get('segments'), list):
            raise RuntimeError('Remote ASR returned no segment timestamps; refusing to invent anchors.')
        return result
