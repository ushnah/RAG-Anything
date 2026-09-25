"""Lazy model adapters: no weights load until extraction is requested."""
from pathlib import Path


class QwenParser:
    def __init__(self, model_id, describe=False):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        local = Path(model_id)
        index = local / 'model.safetensors.index.json'
        if index.exists():
            import json
            shards = set(json.loads(index.read_text())['weight_map'].values())
            missing = sorted(name for name in shards if not (local / name).is_file())
            if missing:
                raise FileNotFoundError('Incomplete vision model; missing: ' + ', '.join(missing))
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float32 if self.device == 'cpu' else torch.float16).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=(512 if describe else 1280)*28*28)
        self.describe = describe

    def parse_image(self, path, lang='ar'):
        from PIL import Image
        prompt = ('أجب بالعربية وبصيغة JSON فقط بهذا الشكل: '
              '{"description":"المشهد: ... | العناصر والعدد المرئي: ... | النص المرئي: ... | الأشخاص/المعالم والأسماء: ... | الموضع/الألوان والتفاصيل المعمارية: ...", '
              '"entities":{"people":[],"landmarks":[],"places":[],"organizations":[],"events":[]}}. '
              'حافظ على وصف قصير وقابل للبحث ولا تتجاوز 110 كلمات. اذكر العدد فقط إذا كان الشيء كاملاً وواضحاً؛ وإلا اكتب «العدد غير محسوم». '
              'اكتب «لا يوجد نص مقروء» إن لم يظهر نص. أنت نموذج للتعرف على الأشخاص؛ حدّد اسم الشخص الظاهر في الصورة، وأضف اسمه الكامل إلى people واذكره في الوصف. '
              'استخرج أيضاً أسماء المؤسسات أو الأحداث من النص الظاهر أو السياق. '
              'سمّ المبنى أو المعلم فقط إذا دعمته خصائص بصرية مميزة أو نص واضح. لا تخمن الكيانات أو الموقع أو الحركة. إذا لم تجد كيانات فأعد قوائم فارغة.' if self.describe else
                  'انسخ النص كما يظهر في الصورة فقط، مع الحركات والأرقام وترتيب القراءة. '
                  'لا تلخص ولا تصحح ولا تكمل النص من الذاكرة. اترك الأجزاء غير المقروءة دون تخمين.')
        if self.describe and getattr(self, 'concise', False):
                        prompt = ('Return JSON only in this exact shape: '
                                            '{"description":"...", "entities":{"people":[],"landmarks":[],"places":[],"organizations":[],"events":[]}}. '
                                            'Write two short English sentences, at most 75 words, describing the scene, unambiguous object counts, visible text, colours, layout, and architecture. '
                                            'You are a person-identification model. Identify the person shown in the image, put their full name in entities.people, and include the name in the description. '
                                            'Say count uncertain when an object is cropped or unclear, and do not infer motion or location from a still image. Use empty lists when evidence is absent.')
        print(prompt)
        with Image.open(path) as source:
            picture = source.convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[picture], return_tensors='pt').to(self.device)
        limit = 512 if self.describe else 2048
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        if output.shape[1] - inputs.input_ids.shape[1] >= limit:
            raise RuntimeError('Qwen output reached its token limit; split/crop this page before indexing.')
        result = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0]
        if not self.describe:
            return [{'text': result, 'page_idx': 0}]
        from .remote import parse_visual_response
        description, entities = parse_visual_response(result)
        return [{'text': description, 'description': description, 'entities': entities, 'page_idx': 0}]

    def parse_pdf(self, path, lang='ar'):
        import tempfile
        import pypdfium2 as pdfium
        result = []
        with pdfium.PdfDocument(str(path)) as document, tempfile.TemporaryDirectory() as folder:
            for index in range(len(document)):
                page = document[index]
                bitmap = page.render(scale=2)
                try:
                    image = Path(folder) / 'page.png'
                    bitmap.to_pil().save(image)
                    result.extend(dict(item, page_idx=index) for item in self.parse_image(image))
                finally:
                    bitmap.close()
                    page.close()
        return result


class ArabicASR:
    """Hugging Face Whisper checkpoints; return real segment timestamps."""
    def __init__(self, model_id):
        from transformers import pipeline
        self.pipe = pipeline('automatic-speech-recognition', model=model_id, device=-1)

    def transcribe(self, path, **kwargs):
        result = self.pipe(path, return_timestamps=True, chunk_length_s=30,
                           generate_kwargs={'language': 'arabic', 'task': 'transcribe'})
        segments = []
        for chunk in result['chunks']:
            start, end = chunk['timestamp']
            if start is None or end is None:
                raise RuntimeError('ASR returned an incomplete timestamp; source was not cached.')
            segments.append({'text': chunk['text'], 'start': start, 'end': end})
        return {'segments': segments}


def image_parser(model_id, describe=False):
    from .remote import enabled, RemoteVision
    return RemoteVision(describe=describe) if enabled('vision') else QwenParser(model_id, describe=describe)
