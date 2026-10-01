"""Reference-face identification and visual-description handoff for Sheikh POC."""
import argparse
import json
import os
from pathlib import Path


def identify_people(image_path, face_identifier):
    """Return canonical matched gallery names and full match provenance."""
    matches = face_identifier.identify_faces(image_path)
    people = list(dict.fromkeys(match['name'] for match in matches
                                if isinstance(match, dict) and match.get('name')))
    return {'people': people}, matches


def describe_with_matches(image_path, vision, face_identifier):
    """Identify gallery matches, pass them to the VLM, and return merged output."""
    result, matches = identify_people(image_path, face_identifier)
    context = ({'people': result['people'], 'person_matches': matches,
                'source': 'face_recognition'} if result['people'] else None)
    item = (vision.parse_image(image_path, person_context=context)[0]
            if context else vision.parse_image(image_path)[0])
    from .remote import merge_person_context
    entities = merge_person_context(item.get('entities', {}), context)
    return {
        'people': result['people'],
        'description': item.get('description', item.get('text', '')),
        'entities': entities,
        'person_matches': matches,
    }


def refresh_saved_results(results_path, face_identifier):
    """Refresh only face matches/entities in a saved visual-lab JSON file."""
    results_path = Path(results_path)
    payload = json.loads(results_path.read_text(encoding='utf-8'))
    rows = payload.get('results')
    if not isinstance(rows, list):
        raise ValueError('Saved results JSON must contain a results list')
    all_people = set(payload.get('people', []))
    for row in rows:
        frame = row.get('frame')
        if not frame:
            continue
        people_result, matches = identify_people(frame, face_identifier)
        names = people_result['people']
        entities = row.get('entities')
        if not isinstance(entities, dict):
            entities = {}
        existing_people = entities.get('people', [])
        if isinstance(existing_people, str):
            existing_people = [existing_people]
        if not isinstance(existing_people, list):
            existing_people = []
        entities['people'] = list(dict.fromkeys(
            [name for name in existing_people if isinstance(name, str)] + names))
        row['entities'] = entities
        row['people'] = names
        row['person_matches'] = matches
        all_people.update(names)
    payload['people'] = sorted(all_people)
    temporary = results_path.with_suffix(results_path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(results_path)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, nargs='?')
    parser.add_argument('--refresh-results', type=Path,
                        help='Update people/entities in saved visual-lab JSON without calling the VLM')
    parser.add_argument('--gallery', type=Path, default=None)
    parser.add_argument('--model', default=None, help='Face model name; defaults to FACE_MODEL or buffalo_l')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--qwen-vl-model', default='models/qwen2.5-vl-3b')
    args = parser.parse_args()
    if not args.image and not args.refresh_results:
        parser.error('provide an image or --refresh-results')
    if args.image and args.refresh_results:
        parser.error('provide either an image or --refresh-results, not both')
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / '.env')
    from .face_identifier import FaceIdentifier
    gallery = args.gallery or Path(os.getenv('FACE_GALLERY_PATH', 'reference_faces'))
    if not gallery.is_absolute():
        gallery = project_root / gallery
    identifier = FaceIdentifier(
        gallery,
        model_name=args.model or os.getenv('FACE_MODEL', 'buffalo_l'),
        threshold=args.threshold if args.threshold is not None else float(os.getenv('FACE_MATCH_THRESHOLD', '0.5')),
        min_face_size=int(os.getenv('MIN_FACE_SIZE', '40')),
        min_detection_score=float(os.getenv('MIN_DETECTION_SCORE', '0.6')),
    )
    if args.refresh_results:
        refreshed = refresh_saved_results(args.refresh_results, identifier)
        print(json.dumps({'people': refreshed['people']}, ensure_ascii=False, indent=2))
        return
    from .models import image_parser
    vision = image_parser(os.getenv('QWEN_VL_MODEL', args.qwen_vl_model), describe=True)
    print(json.dumps(describe_with_matches(args.image, vision, identifier), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
