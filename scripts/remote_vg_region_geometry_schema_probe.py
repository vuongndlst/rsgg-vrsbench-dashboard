"""Bounded reference schema preparation; no captions generated or test scored."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, importlib.util, json, os, resource, shutil, signal, sqlite3, time
root=Path.home()/'rsg-former'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
signal.alarm(75)
resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
started=time.monotonic()
folder=root/'artifacts/caption_readiness/vg_region_geometry_schema_probe_v1'
assert not folder.exists(), 'preserve_existing_probe_no_automatic_rerun'
assert shutil.disk_usage(root).free>16*1024**3+16*1024**2
source=root/'src/rsgformer/vg_caption_region_units.py'
config=root/'configs/vg_region_caption_geometry_v1.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='bb39b56219e2a0ed92fd95ac71e0e8294cdddf3a70a2a7a5f47056b48d810e2a'
assert hashlib.sha256(config.read_bytes()).hexdigest()=='62569e6e710f37104f72c57d8cdb7339aa3cc0385746cbba6c90cd1373705131'
protocol=json.loads(config.read_text())
reference=root/'data/vg_caption/region_descriptions.json'
digest=hashlib.sha256()
with reference.open('rb') as handle:
    for block in iter(lambda:handle.read(1024**2),b''): digest.update(block)
assert digest.hexdigest()==protocol['source_region_json_sha256']
spec=importlib.util.spec_from_file_location('region_geometry_probe',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
journal=root/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite'
db=sqlite3.connect(journal.as_uri()+'?mode=ro',uri=True,timeout=5)
db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
vg_source=json.loads(db.execute('SELECT value FROM provenance WHERE key="source"').fetchone()[0])
assert vg_source['script_sha256']=='5573148955570b19bc9f8e3c6834462c4ac9e2709f299e5b1ba3cc74ea48622c'
dimensions={identity:(width,height) for identity,width,height in db.execute('SELECT vg_id,width,height FROM pixels')}
db.close()
lock={'utc':datetime.now(timezone.utc).isoformat(),'scope':'first2000 reference image records; geometry only on already decoded canonical journal identities; no quality-based sampling',
      'source_hashes':{'adapter':hashlib.sha256(source.read_bytes()).hexdigest(),'geometry_protocol':hashlib.sha256(config.read_bytes()).hexdigest(),
                       'probe':hashlib.sha256((root/'scripts/remote_vg_region_geometry_schema_probe.py').read_bytes()).hexdigest()},
      'reference_sha256':digest.hexdigest(),'pixel_dimension_source':vg_source,
      'resource':{'cpu_threads':1,'alarm_seconds':75,'address_space_bytes':768*1024**2,'image_records_cap':2000,
                  'prefix_text_read_bytes_cap':32*1024**2,'disk_reserve_bytes':16*1024**3},
      'reference_text_evaluation_preparation_only':True,'test_scored':False,'live_v2_modified':False}
folder.mkdir()
lock_raw=json.dumps(lock,indent=2).encode();(folder/'preregistration.json').write_bytes(lock_raw)
decoder=json.JSONDecoder();buffer='';read_characters=0;prefix_bytes=0;images=0;covered=0;raw_regions=0;eligible=0;unverified_regions=0
reasons=Counter();region_fields=Counter();parent_fields=Counter();seen=set()
with reference.open(encoding='utf8') as handle:
    assert handle.read(1)=='['
    while images<2000:
        buffer=buffer.lstrip()
        if buffer.startswith(','):buffer=buffer[1:].lstrip()
        try:
            item,end=decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            chunk=handle.read(1024**2)
            assert chunk,'unexpected_reference_eof'
            read_characters+=len(chunk);prefix_bytes+=len(chunk.encode('utf8'));assert prefix_bytes<=32*1024**2
            buffer+=chunk;assert len(buffer)<=16*1024**2
            continue
        buffer=buffer[end:]
        identity=item.get('id',item.get('image_id'))
        module.positive_identity(identity)
        assert identity not in seen,'duplicate_parent_reference_identity';seen.add(identity)
        parent_fields.update(item.keys());images+=1
        rows=item['regions'];raw_regions+=len(rows)
        for row in rows: region_fields.update(row.keys())
        if identity not in dimensions:
            # Schema/ID integrity still checked, but no geometry certification
            # using unverified image metadata or guessed dimensions.
            ids=set()
            for row in rows:
                rid=module.positive_identity(row['region_id'])
                assert module.positive_identity(row['image_id'])==identity
                assert rid not in ids;ids.add(rid)
                assert all(key in row for key in ('x','y','width','height','phrase'))
            unverified_regions+=len(rows)
            continue
        inputs,refs,rejected=module.prepare_provided_crop_units(identity,dimensions[identity],rows)
        assert set(inputs)==set(refs)
        assert not any('phrase' in value or 'human_references' in value for value in inputs.values())
        covered+=1;eligible+=len(inputs);reasons.update(row['reason'] for row in rejected)
assert eligible+sum(reasons.values())+unverified_regions==raw_regions
result={'utc':datetime.now(timezone.utc).isoformat(),'status':'PARTIAL_REFERENCE_SCHEMA_GEOMETRY_PROBE_PASS',
        'preregistration_sha256':hashlib.sha256(lock_raw).hexdigest(),'source_hashes':lock['source_hashes'],
        'reference_sha256':digest.hexdigest(),'image_records_inspected':images,'geometry_verified_image_records':covered,
        'original_regions_inspected':raw_regions,'eligible_geometry_units':eligible,'rejected_geometry_or_reference_reasons':dict(reasons),
        'units_without_audited_pixel_dimensions':unverified_regions,'observed_parent_fields':dict(parent_fields),
        'observed_region_fields':dict(region_fields),'prefix_text_characters_read':read_characters,'prefix_text_bytes_read':prefix_bytes,
        'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'full_source_schema_and_eligibility_audit_complete':False,'caption_evaluation_admitted':False,
        'reference_text_used_for_model_input':False,'image_pixels_read_or_written':False,'network_bytes':0,
        'model_or_metric_run':False,'benchmark_test_scored':False,'live_v2_modified':False,
        'fingerprint_journal_or_training_release_modified':False,
        'next':'Full bounded schema/eligibility manifest audit after dimension coverage; workload/resource admission and matched dev prompt lock remain required'}
(folder/'outcome.json').write_text(json.dumps(result,indent=2))
(root/'reports/VG_REGION_CAPTION_SCHEMA_PROBE.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
