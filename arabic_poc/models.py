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
        prompt = ('أنشئ وصفاً قصيراً قابلاً للبحث بالعربية: المشهد: ... | العناصر والعدد المرئي: ... | النص المرئي: ... | الموضع/الألوان: ... | تفاصيل معمارية: ... . '
                  'اذكر العدد فقط إذا كان الشيء كاملاً وواضحاً في اللقطة؛ وإلا اكتب «العدد غير محسوم». '
                  'اكتب «لا يوجد نص مقروء» إن لم يظهر نص. لا تتجاوز 110 كلمات ولا تكرر المعلومات. '
                  'لا تخمن أسماء الأشخاص أو الأحداث. اذكر الأشياء والعلاقات المرئية فقط. '
                  'اذكر اسم المعلم فقط إن كان واضحاً ومميزاً، وإلا صف شكله دون تخمين. '
                  'لا تستنتج هوية شخص من وجهه. إذا ظهر اسم مكتوب في لافتة أو تعليق '
                  'فانقله باعتباره نصاً مرئياً، ولا تفترض أنه اسم الشخص الظاهر. '
                  'لا تستنتج الحركة من صورة ثابتة.' if self.describe else
                  'انسخ النص كما يظهر في الصورة فقط، مع الحركات والأرقام وترتيب القراءة. '
                  'لا تلخص ولا تصحح ولا تكمل النص من الذاكرة. اترك الأجزاء غير المقروءة دون تخمين.')
        if self.describe and getattr(self, 'concise', False):
            prompt = ('Describe the scene, visible objects and their exact count only when unambiguous, visible text, colours, layout, and architectural details. '
                      'Use two short English sentences, at most 75 words. Say count uncertain if an object is cropped or unclear. '
                      'Do not identify people, infer motion or guess a location. Do not repeat yourself.')
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
        return [{'text': result, 'page_idx': 0}]

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
