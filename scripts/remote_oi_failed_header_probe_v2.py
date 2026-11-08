"""Separate bounded byte-header diagnosis, not a full pixel retry or scoring."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, signal, urllib.request, struct, io
from PIL import Image
root=Path.home()/'rsg-former'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
signal.alarm(35)
folder=root/'artifacts/caption_readiness/oi_failed_header_probe_v2'
assert not folder.exists(), 'preserve_existing_diagnostic'
assert shutil.disk_usage(root).free>16*1024**3+1024**2
audit=json.loads((root/'reports/OI_EXHAUSTED_SOURCE_AUDIT.json').read_text())
assert audit['preregistration_sha256']=='61717d618eaab7306f7d17c302efdef095943f41fcf0d37a9488eb3b905ec775'
ids=['0224c94fe59bcf46','02b5ac344dea8b8e']
assert set(ids).issubset(row['image_id'] for row in audit['records'])
protocol={'utc':datetime.now(timezone.utc).isoformat(),
          'scope':'classify headers of two decoder-rejected locked TEST objects only',
          'ids':ids, 'source_hashes':{'scripts/remote_oi_failed_header_probe_v2.py':hashlib.sha256((root/'scripts/remote_oi_failed_header_probe_v2.py').read_bytes()).hexdigest()},
          'parent_phase_preregistration_sha256':audit['preregistration_sha256'],
          'resource':{'cpu_threads':1,'alarm_seconds':35,'request_timeout_seconds':10,
                      'requests':2,'bytes_per_request':262144,'total_body_bytes_cap':524288},
          'method':'HTTP Range0..262143 only; reject200 before body read; Pillow header parser only/no load or convert, record container/primary size/frame count; no pixel-index write',
          'reason':'v1 signature JPEG but first4KiB preceded large EXIF APP1; inspect decoded-container header classification without full pixel retry',
          'existing_full_pixel_attempt_cap_unchanged':True,'live_v2_modified':False,
          'benchmark_test_scored':False}
folder.mkdir()
raw=json.dumps(protocol,indent=2).encode()
(folder/'preregistration.json').write_bytes(raw)
records=[]
total=0
for identity in ids:
    source='https://open-images-dataset.s3.amazonaws.com/test/'+identity+'.jpg'
    request=urllib.request.Request(source,headers={'Range':'bytes=0-262143'})
    record={'image_id':identity,'source':source}
    try:
        with urllib.request.urlopen(request,timeout=10) as response:
            record.update(http_status=response.status, content_type=response.headers.get('Content-Type'),
                          content_range=response.headers.get('Content-Range'), etag=response.headers.get('ETag'))
            assert response.status==206 and record['content_range'].startswith('bytes 0-'), 'server_range_contract'
            length=int(response.headers.get('Content-Length','0'))
            assert 0<length<=262144
            body=response.read(length)
            total+=len(body)
            assert len(body)==length and total<=524288
            record.update(header_bytes=len(body), header_sha256=hashlib.sha256(body).hexdigest(),
                          signature_hex=body[:16].hex())
            if body.startswith(b'\x89PNG\r\n\x1a\n'):
                assert body[12:16]==b'IHDR'
                record.update(encoded_format='PNG',header_dimensions=list(struct.unpack('>II',body[16:24])))
            elif body.startswith(b'\xff\xd8\xff'):
                record['encoded_format']='JPEG'
            else:
                record['encoded_format']='unclassified_header'
            with Image.open(io.BytesIO(body)) as header:
                record.update(pillow_header_format=header.format,
                              header_dimensions=list(header.size),
                              header_frame_count=getattr(header,'n_frames',1),
                              header_mode=header.mode)
    except Exception as error:
        record['error']=type(error).__name__+': '+str(error)
    records.append(record)
report={'utc':datetime.now(timezone.utc).isoformat(), 'status':'HEADER_AUDIT_COMPLETE' if all('error' not in row for row in records) else 'HEADER_AUDIT_INCOMPLETE',
        'preregistration_sha256':hashlib.sha256(raw).hexdigest(),'records':records,
        'network_body_bytes':total,'full_image_downloaded':False,'model_or_metric_run':False,
        'reference_captions_read':False,'benchmark_test_scored':False,'live_v2_modified':False,
        'production_reader_or_journals_modified':False,
        'next':'Version a separately preregistered exact decoder compatibility repair; preserve old reader, journal, denominator and per-phase limits'}
(folder/'outcome.json').write_text(json.dumps(report,indent=2))
(root/'reports/OI_FAILED_HEADER_PROBE_V2.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
