"""Describe images and update the active app search indexes (not the LightRAG graph)."""
import argparse
import json, shutil, datetime, tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from arabic_poc.__main__ import IMAGES, Extractor, records, save_json
from arabic_poc.visual import embedding_model, search_text
from arabic_poc import video_search
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--refresh', action='store_true', help='Regenerate descriptions for existing images too')
args=parser.parse_args()
if not args.directory.is_dir(): parser.error('Image directory does not exist')
storage=video_search.STORAGE
visual=json.loads((storage/'visual-index.json').read_text())
combined=json.loads((storage/'index.json').read_text())
known={r['source'] for r in visual['records']}
missing=sorted(p.resolve() for p in args.directory.rglob('*') if p.suffix.lower() in IMAGES and (args.refresh or str(p.resolve()) not in known) and 'visual_samples' not in p.parts)
extractor=Extractor(describe_images=True)
class Descriptions:
 def extract(self,path):
  result=extractor.describe(path)
  yield result['description'],dict(kind='visual_description',entities=result['entities'],person_matches=result['person_matches']),extractor.description_engine()
new={}
for i,path in enumerate(missing,1):
 print(f'Describing {i}/{len(missing)}: {path.name}',flush=True)
 rows=records(path,Descriptions())
 new[path]=rows
 save_json(Path('/tmp/image-description-results')/(rows[0]['id']+'.json'),rows)
if not new:
 print('All images already indexed');raise SystemExit
encoder,model=embedding_model()
assert visual['embedding_model']==combined['embedding_model']==model
new_visual=[dict(r,media_type='image',source_label=r.get('metadata',{}).get('visual_label','')) for rows in new.values() for r in rows]
vectors=encoder.encode([search_text(r) for r in new_visual],batch_size=4,return_dense=True)['dense_vecs']
replaced={str(p) for p in new}
kept=[(r,v) for r,v in zip(visual['records'],visual['vectors']) if r['source'] not in replaced]
visual['records']=[r for r,v in kept]+new_visual
visual['vectors']=[v for r,v in kept]+vectors.tolist()
with tempfile.TemporaryDirectory(dir=storage) as tmp:
 stage=Path(tmp)
 shutil.copytree(storage/'sources',stage/'sources')
 save_json(stage/'visual-index.json',visual)
 for file in (stage/'sources').glob('*.json'):
  doc=json.loads(file.read_text())
  doc['records']=[r for r in doc['records'] if not (r['source'] in replaced and r['anchor']['kind']=='visual_description')]
  save_json(file,doc)
 for path,rows in new.items(): save_json(stage/'sources'/(rows[0]['id'].rsplit('-',1)[0]+'.json'),{'fingerprint':{'refresh':'image-additions'},'records':rows})
 video_search.INPUTS=[str(stage.resolve())]
 all_rows=video_search.collect()
 existing={r['id']:v for r,v in zip(combined['records'],combined['vectors'])}
 needed=[r for r in all_rows if r['id'] not in existing]
 if needed:
  embedded=encoder.encode([r['normalized_text'] for r in needed],batch_size=4,return_dense=True)['dense_vecs']
  existing.update({r['id']:v.tolist() for r,v in zip(needed,embedded)})
 combined.update(records=all_rows,vectors=[existing[r['id']] for r in all_rows])
 backup=storage/('image-additions-backup-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
 backup.mkdir()
 for name in ['index.json','visual-index.json']:shutil.copy2(storage/name,backup/name)
 shutil.copytree(storage/'sources',backup/'sources')
 for file in (stage/'sources').glob('*.json'):
  shutil.copy2(file,storage/'sources'/file.name)
 save_json(storage/'visual-index.json',visual)
 save_json(storage/'index.json',combined)
print(f'Added {len(new)} images; total images: {len({r["source"] for r in all_rows if r["media_type"]=="image"})}',flush=True)
