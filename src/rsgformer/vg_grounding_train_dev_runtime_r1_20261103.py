"""Frozen common paths, hash checks and shared generation contracts. No references."""
from pathlib import Path
import json,os,time
from .vg_color_execution_common_v1 import sha,save,line,now,artifact_bytes
ROOT=Path.home()/'rsg-former';CFG=ROOT/'configs/vg_grounding_train_dev_r1_20261103.json';MANIFEST=ROOT/'configs/vg_grounding_train_dev_source_manifest_r1_20261103.json'
PIPE=ROOT/'artifacts/vg_grounding_train_dev_r1_20261103';PREFLIGHT=ROOT/'artifacts/vg_grounding_train_dev_preflight_r1_20261103'
PARENT=ROOT/'artifacts/vg_focus_repair_v1_20261028';CACHE=ROOT/'artifacts/vg_grounding_dev_prepare_v1_20261102'
def verify_inputs():
 cfg=json.loads(CFG.read_text());manifest=json.loads(MANIFEST.read_text())
 for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,('locked_source_drift',p)
 for base,table in [(PARENT,cfg['parent_files_sha256']),(CACHE,cfg['dev_cache_files_sha256'])]:
  for p,h in table.items():assert sha(base/p)==h,('locked_input_drift',str(base/p))
 assert time.time()<1794670321.2414355
 return cfg,manifest

def prompt_rows(unit):
 from .vg_caption_candidate_prompt_v2 import messages
 from .vg_grounding_controls_v1_20261102 import choice_messages
 old,ex=messages(unit,'N-Z','P1',[],[]);assert not ex
 return {'G0':old,'G2':choice_messages(unit['graph'])}
