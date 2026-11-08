"""Fixed two-ID, separately registered primary-frame pixel compatibility pilot."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, resource, shutil, signal, struct, sys, time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
root=Path.home()/'rsg-former'
sys.path.insert(0,str(root/'src'))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
signal.alarm(75)
started=time.monotonic()
folder=root/'artifacts/caption_readiness/oi_mpo_repair_pilot_v1'
assert not folder.exists(), 'preserve_existing_compatibility_version_no_automatic_retry'
assert shutil.disk_usage(root).free>16*1024**3+16*1024**2
reader_sha='8f484a9b98683102a77f0cb4a3d85fb246a6670e63f7b6841261a00b9e915555'
assert hashlib.sha256((root/'src/rsgformer/openimages_test_reader_v2.py').read_bytes()).hexdigest()==reader_sha
from rsgformer.openimages_test_reader_v2 import OpenImagesTestReaderV2
probe=json.loads((root/'reports/OI_READER_V2_CPU_PROBE.json').read_text())
assert probe['status']=='PASS' and probe['reader_v2_sha256']==reader_sha and probe['network_downloaded_bytes']==0
headers=json.loads((root/'reports/OI_FAILED_HEADER_PROBE_V2.json').read_text())
assert headers['status']=='HEADER_AUDIT_COMPLETE' and all(row['pillow_header_format']=='MPO' for row in headers['records'])
ids=['0224c94fe59bcf46','02b5ac344dea8b8e']
assert sorted(row['image_id'] for row in headers['records'])==ids
header_records={row['image_id']:row for row in headers['records']}
protocol={'utc':datetime.now(timezone.utc).isoformat(),
          'scope':'two fixed TEST MPO primary-frame compatibility objects only; no model or scoring',
          'source_hashes':{'scripts/remote_oi_mpo_repair_pilot.py':hashlib.sha256((root/'scripts/remote_oi_mpo_repair_pilot.py').read_bytes()).hexdigest(),
                           'src/rsgformer/openimages_test_reader_v2.py':reader_sha},
          'parent_phase_preregistration_sha256':'61717d618eaab7306f7d17c302efdef095943f41fcf0d37a9488eb3b905ec775',
          'ids':ids,'resource':{'cpu_threads':1,'alarm_seconds':75,'peak_rss_gate_bytes':768*1024**2,
                               'single_request_bytes':8*1024**2,'network_body_bytes':16*1024**2,'requests':2},
          'decode_policy':'JPEG/MPO default primary frame0; raw orientation, no transpose/rotation/resize; hard16Mprimary pixels before decode; no alternate-frame selection',
          'method':'raw Pillow RGB/no EXIF; SHA256(width,height,RGB); normalized DCT32 low8 median excluding DC phash63; 0/90/180/270 rotations',
          'old_phase_attempt_charges_and_journal_preserved':True,
          'supplemental_scope_only_not_full_protected_coverage':True,
          'reference_or_score_selection':False,'live_v2_modified':False,'benchmark_test_scored':False}
folder.mkdir()
raw=json.dumps(protocol,indent=2).encode()
(folder/'preregistration.json').write_bytes(raw)
reader=OpenImagesTestReaderV2(Path.home()/'shared-assets/datasets/openimages_test',
    manifest_sha256='6eb586b40bd68245e20205d896a5ae49700b72d61783b0e5ddc80a50e2242c87',
    allow_network=True,single_image_bytes_cap=8*1024**2,network_bytes_cap=16*1024**2)
basis=np.cos(np.pi/32*np.arange(8)[:,None]*(np.arange(32)[None,:]+.5))
basis*=np.sqrt(2/32);basis[0]/=np.sqrt(2)
records=[]
errors=[]
for identity in ids:
    item=None
    try:
        item=reader.read(identity)
        assert item.provenance['encoded_format']=='MPO' and item.provenance['selected_frame']==0
        assert list(item.image.size)==header_records[identity]['header_dimensions']
        rgb=item.image
        # Same explicitly pinned LANCZOS method as all existing journals.
        from PIL import Image
        gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
        hashes=[]
        for rotation in range(4):
            low=(basis@np.rot90(gray,rotation)@basis.T).reshape(-1)[1:]
            hashes.append(f'{sum(int(value)<<bit for bit,value in enumerate(low>np.median(low))):016x}')
        records.append({'image_id':identity,'jpeg_sha':item.provenance['sha256'],
                        'rgb_sha':hashlib.sha256(struct.pack('>II',rgb.width,rgb.height)+rgb.tobytes()).hexdigest(),
                        'width':rgb.width,'height':rgb.height,
                        'phash0':hashes[0],'phash90':hashes[1],'phash180':hashes[2],'phash270':hashes[3],
                        'provenance':item.provenance})
    except Exception as error:
        errors.append({'image_id':identity,'error':type(error).__name__+': '+str(error)})
    finally:
        if item is not None:item.image.close()
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert peak<768*1024**2 and reader.downloaded_bytes<=16*1024**2
evidence={'protocol':protocol,'records':records,'errors':errors}
evidence_raw=json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()
(folder/'pixel_fingerprints.json').write_bytes(evidence_raw)
report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if len(records)==2 and not errors else 'INCOMPLETE_COMPATIBILITY_PILOT',
        'preregistration_sha256':hashlib.sha256(raw).hexdigest(),'reader_sha256':reader_sha,
        'indexed_images':len(records),'target_images':2,'network_body_bytes':reader.downloaded_bytes,
        'evidence_artifact_sha256':hashlib.sha256(evidence_raw).hexdigest(),
        'evidence_artifact_alias':'${REMOTE_HOME}/rsg-former/artifacts/caption_readiness/oi_mpo_repair_pilot_v1/pixel_fingerprints.json',
        'peak_cpu_rss_bytes':peak,'elapsed_seconds':time.monotonic()-started,'errors':errors,
        'old_journal_or_reader_modified':False,'live_v2_modified':False,'reference_captions_read':False,
        'model_or_metric_run':False,'benchmark_test_scored':False,'disk_pixel_bytes_written':0,
        'full_protected_pixel_coverage':False,'clean_release_certified':False,
        'next':'Preserve supplemental fingerprints; audit separately versioned union/remaining malformed-source coverage before complete-set inference; v1 parent outcome still reports its original failures'}
(folder/'outcome.json').write_text(json.dumps(report,indent=2))
(root/'reports/OI_MPO_REPAIR_PILOT.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
