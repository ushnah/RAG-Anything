"""Generate grounded answers from the evidence returned by library search."""
from .prompt_loader import get_prompt
import json
import os

from .__main__ import checked_answer
from .remote import chat, enabled

SYSTEM = get_prompt('answers.grounded')


def generate(messages):
    if enabled('text'):
        return chat(messages, json_output=True)
    from openai import OpenAI
    with OpenAI(base_url=os.environ['QWEN_BASE_URL'], api_key=os.getenv('QWEN_API_KEY','local'), timeout=180, max_retries=0) as client:
        result = client.chat.completions.create(model=os.environ['QWEN_MODEL'], messages=messages,
            temperature=0, max_tokens=int(os.getenv('QWEN_MAX_TOKENS','1536')), response_format={'type':'json_object'})
        if result.choices[0].finish_reason == 'length':
            raise RuntimeError('Answer exceeded the output limit')
        return result.choices[0].message.content or ''


def answer_from_evidence(question, result, catalog):
    if result.get('no_match') or not result.get('sources'):
        return dict(result, generation='no_match')
    by_id = {row['id']: row for row in catalog}
    selected, parents = {}, {}
    for group in result['sources']:
        candidates = [group] + group.get('matches', [])
        for match in group.get('matches', []):
            candidates += match.get('aligned', [])
        # A publisher label can identify an image, while its generated visual
        # description carries count, colour and placement details. Include that
        # paired evidence for image answers without expanding video context to
        # every frame or transcript segment.
        if group.get('media_type') == 'image':
            candidates += [row for row in catalog if row.get('source') == group.get('source')
                           and row.get('evidence_type') == 'visual_description']
        for candidate in candidates:
            row = by_id.get(candidate['id'])
            if row and row['id'] not in selected and len(selected) < 24:
                selected[row['id']] = row
                parents[row['id']] = group['id']
    evidence = list(selected.values())
    payload = {'question':question,'evidence':[{key:row.get(key) for key in
        ('id','document','original_text','anchor','engine')} for row in evidence]}
    messages = [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
    fallback = dict(result, answer='تعذر توليد إجابة موثقة. يمكنك مراجعة المصادر المسترجعة أدناه.', citations=[], generation='unavailable')
    if not evidence:
        return fallback
    try:
        for attempt in range(2):
            raw = generate(messages)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get('insufficient_evidence') is True and not parsed.get('citations'):
                    return dict(result, answer='الأدلة المسترجعة لا تكفي للإجابة عن هذا السؤال بثقة.', citations=[], generation='insufficient_evidence')
            except ValueError:
                pass
            checked = checked_answer(raw, evidence)
            if checked['citations']:
                for citation in checked['citations']:
                    citation['source_id'] = parents[citation['id']]
                    citation['anchor'] = selected[citation['id']]['anchor']
                    citation['evidence_type'] = citation['anchor']['kind']
                return dict(result, **checked, generation='grounded')
            messages.append({'role':'user','content':get_prompt('answers.grounded_repair')})
    except Exception:
        # Preserve useful retrieval even if the generation service is unavailable.
        return fallback
    return dict(fallback, generation='invalid_citations')
