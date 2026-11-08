"""Invented JPEG/MPO compatibility contracts, no real TEST download."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, io, json, os, resource, signal, sys, time
from PIL import Image
root=Path.home()/'rsg-former'
sys.path.insert(0,str(root/'src'))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
signal.alarm(60)
started=time.monotonic()
old_sha='0bf8d41c541109e11c89e3e7dbaf47b38234a87a995baf2e5f2f2e0b16d307f6'
new_sha='8f484a9b98683102a77f0cb4a3d85fb246a6670e63f7b6841261a00b9e915555'
for name, expected in [('openimages_test_reader.py',old_sha),('openimages_test_reader_v2.py',new_sha)]:
    assert hashlib.sha256((root/'src/rsgformer'/name).read_bytes()).hexdigest()==expected
from rsgformer.openimages_test_reader import OpenImagesTestReader
from rsgformer.openimages_test_reader_v2 import OpenImagesTestReaderV2
folder=root/'artifacts/caption_readiness/oi_reader_v2_cpu_fixture_v1'
assert not folder.exists(), 'preserve_existing_fixture'
folder.mkdir()
(folder/'images').mkdir()
protocol={'utc':datetime.now(timezone.utc).isoformat(),'scope':'invented encoded-image compatibility only',
          'reader_v1_sha256':old_sha,'reader_v2_sha256':new_sha,
          'resource':{'cpu_threads':1,'alarm_seconds':60,'peak_rss_gate_bytes':512*1024**2},
          'cases':['ordinary JPEG exact parity','MPO default primary frame0','large primary shape retained without resize','over16Mpixel refusal','PNG refusal'],
          'network_allowed':False,'real_test_pixels_accessed':False,'live_v2_modified':False}
(folder/'preregistration.json').write_text(json.dumps(protocol,indent=2))
identities=[f'{index:016x}' for index in range(1,6)]
manifest={'caption_test_ids':identities,'images':[{'ImageID':identity,'Rotation':0} for identity in identities]}
raw=json.dumps(manifest).encode()
(folder/'test_identity_manifest.json').write_bytes(raw)
manifest_sha=hashlib.sha256(raw).hexdigest()
def save_image(index,size,color,format='JPEG'):
    image=Image.new('RGB',size,color)
    image.save(folder/'images'/(identities[index]+'.jpg'),format=format)
    image.close()
save_image(0,(17,25),(40,80,120))
primary=Image.new('RGB',(120,60),(200,30,20))
secondary=Image.new('RGB',(120,60),(10,30,220))
primary.save(folder/'images'/(identities[1]+'.jpg'),format='MPO',save_all=True,append_images=[secondary])
primary.close();secondary.close()
save_image(2,(1025,32),(25,80,40))
save_image(3,(4001,4000),(1,2,3))
save_image(4,(17,25),(40,80,120),format='PNG')
old=OpenImagesTestReader(folder,manifest_sha256=manifest_sha)
new=OpenImagesTestReaderV2(folder,manifest_sha256=manifest_sha)
first=old.read(identities[0]);second=new.read(identities[0])
assert first.image.size==second.image.size and first.image.tobytes()==second.image.tobytes()
assert first.provenance['sha256']==second.provenance['sha256']
first.image.close();second.image.close()
mpo=new.read(identities[1])
assert mpo.provenance['encoded_format']=='MPO' and mpo.provenance['encoded_frame_count']==2 and mpo.provenance['selected_frame']==0
with Image.open(folder/'images'/(identities[1]+'.jpg')) as source:
    reference=source.convert('RGB')
    assert reference.tobytes()==mpo.image.tobytes()
    reference.close()
mpo.image.close()
large=new.read(identities[2])
assert large.image.size==(1025,32) and large.provenance['official_object_longest_side_exceeds_nominal1024']
large.image.close()
for identity,error_message in [(identities[3],'test_primary_frame_pixel_cap_exceeded'),(identities[4],'unexpected_test_pixel_format')]:
    try:new.read(identity)
    except ValueError as error:assert str(error)==error_message
    else:raise AssertionError('guard_did_not_refuse:'+identity)
try:new.read('f'*16)
except ValueError as error:assert str(error)=='identity_not_in_locked_test_manifest'
else:raise AssertionError('non_test_identity_accepted')
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert peak<512*1024**2
report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
        'reader_v1_sha256':old_sha,'reader_v2_sha256':new_sha,'cases':protocol['cases']+['non-test identity refusal'],
        'elapsed_seconds':time.monotonic()-started,'peak_cpu_rss_bytes':peak,
        'network_downloaded_bytes':0,'real_test_pixels_accessed':False,
        'invented_fixture_pixels_written':True,'production_reader_or_journals_modified':False,
        'model_or_metric_run':False,'benchmark_test_scored':False,'live_v2_modified':False,
        'actual_rejected_test_objects_fully_decoded':False,
        'next':'Separately register bounded actual MPO primary-frame repair and supplemental journal; do not overwrite live v1 phase/limits or omit failed identities'}
(folder/'outcome.json').write_text(json.dumps(report,indent=2))
(root/'reports/OI_READER_V2_CPU_PROBE.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
