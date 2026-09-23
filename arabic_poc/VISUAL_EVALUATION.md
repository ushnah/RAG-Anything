# Makkah / Madinah visual evaluation

## Scope and reproducibility

Four new public videos in `data/visual_evaluation/`, totalling about 226 seconds, are sampled every 20 seconds into 13 frames. Three short Madinah clips contribute one frame each; this is not a full-video coverage audit. The configured vision backend generates descriptions and optional structured entities; BGE-M3 retrieves video frames from the generated descriptions and entity names.

```sh
.venv311/bin/python -m arabic_poc.visual --storage rag_storage_arabic_visual_eval build data/visual_evaluation --frame-seconds 20
.venv311/bin/python -m arabic_poc.evaluate_visual
```

On this Mac, Qwen inference required GPU access outside the tool sandbox. This evaluation does not run Whisper or LightRAG graph retrieval: it tests the separate visual-search path. No model was trained or fine-tuned.

## Sources

| File | Publisher / attribution | License |
| --- | --- | --- |
| `makkah_timelapse.webm` | [Makkah and Hajj time-lapse](https://commons.wikimedia.org/wiki/File:Time_lapse_of_Masjid_al-Ḥarām_(kaaba)_&_hajj_rites.webm), Masajida Allah | CC BY 3.0 as recorded by Commons |
| `madinah_interior.mp4` | [Mosque interior](https://www.pexels.com/video/aerial-view-of-worship-in-medina-mosque-35189228/), Earth Photart | Pexels License |
| `madinah_umbrellas.mp4` | [Courtyard umbrellas](https://www.pexels.com/video/crowd-at-al-masjid-an-nabawi-in-medina-31486706/), Aamir Somewhere | Pexels License |
| `madinah_dome.mp4` | [Green dome](https://www.pexels.com/video/iconic-green-dome-and-minarets-of-medina-mosque-35940331/), Asad Ansari | Pexels License |

Downloads are unchanged. Metadata sidecars preserve source links, authors, licenses, hashes, and durations. Geographic labels follow publishers and are not independent geolocation verification. Frame JPEGs in `validation/visual_eval_frames/` are resized extractions from these videos and retain the corresponding source attribution/license.

## Evaluation design

- Provisional reference objects were annotated by the assistant from the actual sampled frames before reviewing generated descriptions. A person should independently review these labels before reporting benchmark accuracy to a client.
- Ten positive queries cover Arabic/English dome searches, umbrellas, a person in a blue shirt, arches, columns, Kaaba, crowds, mountains, and chairs.
- Three queries lack verified support: airplane, red car, and a named person. There are no identity labels, so the last query cannot measure face-recognition accuracy.
- Retrieval is measured with publisher labels and again using descriptions alone. Labels are not provided to the vision model.
- Report Hit@1, Hit@3, Recall@3, and MRR across positive queries. No threshold is tuned on this set. This small, related corpus is not a held-out production benchmark.
- The current ranker always returns nearest candidates. Returning candidates for unsupported queries reveals the lack of an abstention decision; it is not a positive identification.
- Presence in a description is not object detection with bounding boxes. Sampled frames do not establish temporal tracking or action recognition.

See [reference frames and labels](validation/visual_eval_labels.json) and [retrieval results](validation/visual_eval_retrieval.json).

## Results

| Retrieval variant | Hit@1 | Hit@3 | Recall@3 | MRR |
| --- | --- | --- | --- | --- |
| Current pipeline: descriptions + source labels | 60% | 90% | 79.4% | 0.742 |
| Descriptions only | 70% | 90% | 79.4% | 0.825 |

These metrics cover ten positive queries over the 13 new frames, not the expanded demo corpus. Similar frames and repeated scenes make this easier than a large archive. No inference of statistical significance is warranted.

### What worked

- Arabic green-dome, umbrella, arch, Kaaba, crowd, and English mountain queries retrieved relevant frames first in the current pipeline.
- Qwen recognized the green dome/minarets, people in most scenes, mountains in some views, and Kaaba in the two aerial frames.

### What failed

- The person-in-blue-shirt query missed the relevant frame in the top three in both variants. Qwen omitted the shirt color from its description.
- English green-dome search ranked an unrelated hill/crowd scene first; Qwen had invented green terrain/text in that scene.
- Qwen called courtyard umbrellas columns, omitted chairs, confused sitting with running in the interior, and invented other scene details.
- Three of 13 descriptions contain Chinese characters despite the Arabic prompt.
- Provisional semantic review found mentions corresponding to 20 of 31 reference object occurrences. This is descriptive coverage only, not bounding-box detection accuracy or precision; unsupported claims are recorded separately.
- Every unsupported query returned candidates in both variants. The ranker has no calibrated no-match decision. The named-person query has no verified identity evidence and is not a face-recognition test.

Full source descriptions and review notes: [visual_eval_description_review.json](validation/visual_eval_description_review.json).

## Demo

The 13 new records were merged with the five existing records into `rag_storage_arabic_visual/visual-index.json`. The prior index is retained beside it as `visual-index.before-makkah-madinah.json`. Refresh the UI and choose **Images & scenes → Video only**. Try:

- `أرني القبة الخضراء`
- `أرني المظلات الكبيرة في الساحة`
- `أرني الكعبة من الأعلى`
- `Show me a person wearing a blue shirt` (known failure)

## Next evaluation priorities

1. Independently review reference frames and annotations.
2. Compare a stronger vision model on these same frames, without changing labels or hiding failures.
3. Add explicit object attributes and test direct visual embeddings; captions can omit the very detail a query asks for.
4. Add and calibrate a no-match decision on a separate validation set.
5. Increase sampling coverage and evaluate temporal activities separately; integrate speech only where relevant.

GPT-OSS gateway check during this run: corrected the model alias to `cerebras/gpt-oss-120b`; the Arabic text test returned HTTP 200 and `مريم.`. This did not change the local vision model or the demo's model configuration.
