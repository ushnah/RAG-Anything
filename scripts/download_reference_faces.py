"""Download licensed Wikimedia Commons candidates for the face gallery."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_PEOPLE = {
    'abdulaziz_ibn_saud': 'Ibn Saud portrait',
    'abdul_rahman_al_sudais': 'Abdul Rahman Al-Sudais',
    'saleh_al_fawzan': 'Saleh Al-Fawzan',
    'abdulaziz_ibn_baz': 'Abdul Aziz ibn Baz',
    'muhammad_ibn_uthaymeen': 'Muhammad ibn al-Uthaymeen',
    'salman_bin_abdulaziz': 'Salman bin Abdulaziz Al Saud',
    'mohammed_bin_salman': 'Mohammed bin Salman',
}
API = 'https://commons.wikimedia.org/w/api.php'
USER_AGENT = 'RAG-Anything reference gallery downloader/1.0'


def request_json(params):
    request = Request(API + '?' + urlencode(params), headers={'User-Agent': USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def search_person(name, limit):
    payload = request_json({
        'action': 'query', 'format': 'json', 'generator': 'search',
        'gsrsearch': name, 'gsrnamespace': 6, 'gsrlimit': limit,
        'prop': 'imageinfo', 'iiprop': 'url|size|extmetadata', 'iiurlwidth': 800,
    })
    pages = payload.get('query', {}).get('pages', {}).values()
    return [page for page in pages if page.get('imageinfo')]


def safe_extension(url):
    match = re.search(r'\.(jpe?g|png|webp)(?:$|\?)', url, re.IGNORECASE)
    return '.' + match.group(1).lower().replace('jpeg', 'jpg') if match else '.jpg'


def download_person(gallery, folder, name, limit):
    target = gallery / folder
    target.mkdir(parents=True, exist_ok=True)
    metadata_path = target / 'sources.json'
    metadata = []
    for index, page in enumerate(search_person(name, limit), start=1):
        info = page['imageinfo'][0]
        url = info.get('thumburl') or info.get('url')
        if not url:
            continue
        image_path = target / f'{index:03d}{safe_extension(url)}'
        request = Request(url, headers={'User-Agent': USER_AGENT})
        with urlopen(request, timeout=60) as response:
            image_path.write_bytes(response.read())
        ext = info.get('extmetadata', {})
        metadata.append({
            'file': image_path.name,
            'title': page.get('title'),
            'source_url': info.get('descriptionurl'),
            'image_url': info.get('url'),
            'license': ext.get('LicenseShortName', {}).get('value'),
            'artist': ext.get('Artist', {}).get('value'),
        })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{folder}: downloaded {len(metadata)} candidate(s)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, default=Path('reference_faces'))
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--person', action='append', metavar='FOLDER=SEARCH_NAME',
                        help='Override defaults or add a person; repeatable')
    args = parser.parse_args()
    people = dict(DEFAULT_PEOPLE)
    for value in args.person or []:
        folder, separator, name = value.partition('=')
        if not separator or not folder or not name:
            parser.error('--person must use FOLDER=SEARCH_NAME')
        people[folder] = name
    args.gallery.mkdir(parents=True, exist_ok=True)
    for folder, name in people.items():
        try:
            download_person(args.gallery, folder, name, args.limit)
        except Exception as exc:
            print(f'{folder}: download failed: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()
