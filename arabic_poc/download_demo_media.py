"""Download small, openly licensed additions for the Arabic RAG demo.

The source URL, author, licence, checksum, and any local transformation are
written beside every file so the UI can show attribution.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from .__main__ import save_json

COMMONS = 'https://commons.wikimedia.org/w/api.php?'
SAMPLES = [
    {
        'title': 'File:Register on the WikiLearn platform - Arabic version.webm',
        'name': 'wikilearn_arabic.webm',
        'author': 'Mounir Neddi',
        'license': 'CC0 1.0',
        'license_url': 'https://creativecommons.org/publicdomain/zero/1.0/',
        'label': 'شرح عربي للتسجيل في منصة ويكي تعلم',
        'download_url': 'https://upload.wikimedia.org/wikipedia/commons/7/78/Register_on_the_WikiLearn_platform_-_Arabic_version.webm',
    },
    {
        'title': 'File:A Step by Step Guide to Hajj (Islamic pilgrimages).webm',
        'name': 'hajj_guide.webm',
        'author': 'Darussalam Publishers & Distributors',
        'license': 'CC BY 3.0',
        'license_url': 'https://creativecommons.org/licenses/by/3.0/',
        'label': 'دليل لمناسك الحج في مكة المكرمة',
        'download_url': 'https://upload.wikimedia.org/wikipedia/commons/e/e9/A_Step_by_Step_Guide_to_Hajj_%28Islamic_pilgrimages%29.webm',
    },
    {
        'title': 'File:Green Dome, Masjid Al Nabawi.jpg',
        'name': 'green_dome_madinah.jpg',
        'author': 'Wadie Al-Houh',
        'license': 'CC0 1.0',
        'license_url': 'https://creativecommons.org/publicdomain/zero/1.0/',
        'label': 'القبة الخضراء في المسجد النبوي بالمدينة المنورة',
        'download_url': 'https://upload.wikimedia.org/wikipedia/commons/d/d3/Green_Dome%2C_Masjid_Al_Nabawi.jpg',
    },
    {
        'title': 'File:Haram minaret.jpg',
        'name': 'haram_minarets_makkah.jpg',
        'author': 'Yousefmadari',
        'license': 'Public domain',
        'license_url': 'https://commons.wikimedia.org/wiki/File:Haram_minaret.jpg',
        'label': 'مآذن مضاءة عند بوابة الملك فهد في المسجد الحرام بمكة',
        'download_url': 'https://upload.wikimedia.org/wikipedia/commons/6/64/Haram_minaret.jpg',
    },
]


def media_url(title):
    query = urlencode({'action': 'query', 'format': 'json', 'prop': 'imageinfo',
                       'iiprop': 'url', 'titles': title})
    with urlopen(COMMONS + query, timeout=60) as response:
        pages = json.load(response)['query']['pages']
    page = next(iter(pages.values()))
    return page['imageinfo'][0]['url']


def download(entry, destination):
    target = destination / entry['name']
    source = entry.get('download_url') or media_url(entry['title'])
    if not target.exists():
        with urlopen(source, timeout=180) as response, target.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    save_json(target.with_name(target.name + '.metadata.json'), {
        'title': entry['title'][5:], 'source_url': 'https://commons.wikimedia.org/wiki/' + entry['title'].replace(' ', '_'),
        'download_url': source, 'author': entry['author'], 'license': entry['license'],
        'license_url': entry['license_url'], 'visual_label': entry['label'],
        'label_provenance': 'Publisher file description; applies to the whole asset, not individual frames.',
        'sha256': digest, 'transformations': 'None; downloaded original public media.',
    })
    print(f'{target} {digest}')


def main():
    folder = Path('data/demo_additions')
    folder.mkdir(parents=True, exist_ok=True)
    for entry in SAMPLES:
        download(entry, folder)


if __name__ == '__main__':
    main()
