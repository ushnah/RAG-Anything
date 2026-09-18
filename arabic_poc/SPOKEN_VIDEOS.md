# Spoken Arabic videos for the demo

The **Spoken Arabic** collection adds actual speech video to the local UI.

| Demo file | Content | Duration |
| --- | --- | --- |
| `data/spoken_videos/wikidata_tunisian_lecture_60_105s.mp4` | Tunisian-Arabic tutorial lecture by Houcemeddine Turki on extracting information with Wikidata Query Service | 45 seconds, from 01:00–01:45 of the original |
| `data/spoken_videos/david_syrian_arabic.mp4` | David introduces himself in Syrian Arabic and discusses education, hobbies, and language | Full recording, approximately 43 seconds |

## Sources and attribution

- Lecture: [WikiConference RU — Wikidata Query Service Tutorial in Tunisian — Part 1](https://commons.wikimedia.org/wiki/File:WikiConference_RU_-_Wikidata_Query_Service_Tutorial_in_Tunisian_-_Part_1.webm). Speaker: Houcemeddine Turki. The complete 24-minute video is retained as `data/media_originals/wikidata_tunisian_lecture_full.webm`. Its license could not be independently retrieved during this run; the sidecar records this explicitly rather than asserting a license.
- Speaker: [WIKITONGUES — David speaking Syrian Arabic](https://commons.wikimedia.org/wiki/File:WIKITONGUES-_David_speaking_Syrian_Arabic.webm). Wikitongues / Tushar Rakheja, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Original retained as `data/media_originals/david_syrian_arabic_full.webm`. [Community Arabic subtitles](https://commons.wikimedia.org/wiki/TimedText:WIKITONGUES-_David_speaking_Syrian_Arabic.webm.ar.srt) provide an independent comparison for the speech content.

Demo copies are transcoded to 480p H.264/AAC MP4 for browser playback. Each has a `.metadata.json` sidecar with attribution, download URL, SHA-256, transformations, language, and original start time.

## Extraction mode

These videos use `--speech-only`: Whisper-small transcribes their audio, adjacent speech segments are grouped into windows targeting at most 20 seconds, and video-frame OCR is skipped. An individual ASR segment longer than 20 seconds remains intact. This avoids indexing noisy screen text from the lecture's mixed-language software interface. It does not mute or remove the video's pictures; the UI still plays the full demo video at each speech timestamp.

Existing text and media collections retain their original extraction settings. The new corpus lives in `rag_storage_arabic_spoken`.

```sh
.venv311/bin/python -m arabic_poc --storage rag_storage_arabic_spoken extract data/spoken_videos --speech-only
RAG_TOKENIZER_MODEL=models/qwen2.5-vl-3b .venv311/bin/python -m arabic_poc --storage rag_storage_arabic_spoken ingest data/spoken_videos --speech-only
```

## Demo questions

Choose **Spoken Arabic** in the web UI:

- ما هوايات المتحدث في الفيديو؟
- ما الخدمة التي يذكرها المتحدث لاستخراج المعطيات من ويكي بيانات؟

For lecture citations, add **60 seconds** to the clip timestamp to locate the passage in the original recording.

## Verification and limits

Both files have video and audio streams. Extraction produced three speech windows per video, with all timestamps inside file duration. The lecture transcript contains discussion of Wikidata and extracting data; the Syrian transcript includes education, sports, piano, and Arabic as a first language. Full extracted passages are in [validation/spoken_videos_extraction.json](validation/spoken_videos_extraction.json).

These are automatic transcripts, not corrected reference transcripts. Whisper-small makes noticeable errors on dialect, names, and technical terms. This is a demo-data addition, not a claim that the generation and faithfulness failures documented in MEDIA_VALIDATION.md are solved.

All six speech passages reached `processed` status in the new index; see [validation/spoken_videos_index.json](validation/spoken_videos_index.json). The local web server has been refreshed to expose the collection. Twelve POC checks (including speech-window grouping without frame OCR) and five HTTP checks passed.

A live web-UI question about the speaker's hobbies completed in 52.2 seconds. Retrieval included the relevant 20–38-second passage, but Qwen cited the unrelated 38–41-second closing passage. The new corpus is available for the demo; **citation relevance still fails** in this example. See [validation/spoken_videos_query_review.json](validation/spoken_videos_query_review.json). No generation-quality improvement is claimed by adding these videos.
