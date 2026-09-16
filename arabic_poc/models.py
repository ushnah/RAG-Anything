"""Lazy model adapters: no weights load until extraction is requested."""
from pathlib import Path


class QwenParser:
    def __init__(self, model_id, describe=False):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float32 if self.device == 'cpu' else torch.float16).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=1280*28*28)
        self.describe = describe

    def parse_image(self, path, lang='ar'):
        from PIL import Image
        prompt = ('صف محتوى الصورة بالعربية وصفاً دقيقاً للبحث. لا تخمن أسماء الأشخاص أو الأحداث. '
                  'اذكر الأشياء والعلاقات المرئية فقط.' if self.describe else
                  'انسخ النص كما يظهر في الصورة فقط، مع الحركات والأرقام وترتيب القراءة. '
                  'لا تلخص ولا تصحح ولا تكمل النص من الذاكرة. اترك الأجزاء غير المقروءة دون تخمين.')
        with Image.open(path) as source:
            picture = source.convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[picture], return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        if output.shape[1] - inputs.input_ids.shape[1] >= 2048:
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
