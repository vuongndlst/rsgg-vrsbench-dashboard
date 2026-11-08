"""Read-only exact hash comparison of partial protected pixel journals.

This records evidence only: no manifest changes, perceptual decisions or scores.
"""
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import resource
import shutil
import signal
import sqlite3
import time


def protected_internal(source_split, target_split):
    return ((source_split == 'train' and target_split in ('val', 'test'))
            or (source_split == 'val' and target_split == 'test'))


def snapshot(path, columns, ordering, expected_script):
    # mode=ro and query_only prevent accidental mutation of growing journals.
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        source_raw = db.execute('SELECT value FROM provenance WHERE key="source"').fetchone()[0]
        source = json.loads(source_raw)
        assert source['script_sha256'] == expected_script, 'journal_method_source_changed'
        rows = db.execute('SELECT ' + columns + ' FROM pixels ORDER BY ' + ordering).fetchall()
    digest = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    return source, rows, digest


def merge_protected_rows(local_rows, stream_rows):
    """Same identity cannot acquire inconsistent fingerprints across readers."""
    merged = {}
    duplicates = 0
    for collection in (local_rows, stream_rows):
        for row in collection:
            key = tuple(row[:3])
            assert row[1] in ('val', 'test') and len(row) == 7
            if key in merged:
                assert tuple(merged[key]) == tuple(row), 'protected_identity_fingerprint_conflict'
                duplicates += 1
            else:
                merged[key] = tuple(row)
    return [merged[key] for key in sorted(merged)], duplicates


def main():
    started = time.monotonic()
    signal.alarm(60)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 ** 2, 768 * 1024 ** 2))
    root = Path.home() / 'rsg-former'
    folder = root / 'artifacts/caption_readiness'
    assert shutil.disk_usage(root).free > 16 * 1024 ** 3 + 64 * 1024 ** 2
    expected_vg = '5573148955570b19bc9f8e3c6834462c4ac9e2709f299e5b1ba3cc74ea48622c'
    expected_external = '4b254dc19d2be433f42f5cc08a2797ff5d443d871f5f38028792d71b2628d524'
    expected_stream = '752b93fc302851e5e1922bf0a918842f09fca29eade1136cf2b65a4d69f0e1dd'
    for name, expected in [('remote_vg_pixel_fingerprint_batch.py', expected_vg),
                           ('remote_external_pixel_fingerprint_batch.py', expected_external),
                           ('remote_oi_stream_fingerprint_batch.py', expected_stream)]:
        assert hashlib.sha256((root / 'scripts' / name).read_bytes()).hexdigest() == expected
    vg_source, vg_rows, vg_digest = snapshot(
        folder / 'vg_pixel_fingerprints_v1.sqlite',
        'vg_id,split,jpeg_sha,rgb_sha,width,height', 'vg_id', expected_vg)
    external_source, external_rows, external_digest = snapshot(
        folder / 'external_pixel_fingerprints_v1.sqlite',
        'dataset,split,image_id,jpeg_sha,rgb_sha,width,height',
        'dataset,split,image_id', expected_external)
    stream_source, stream_raw_rows, stream_digest = snapshot(
        folder / 'oi_stream_fingerprints_full_v1.sqlite',
        'image_id,jpeg_sha,rgb_sha,width,height', 'image_id', expected_stream)
    assert stream_source['preregistration_sha256'] == '61717d618eaab7306f7d17c302efdef095943f41fcf0d37a9488eb3b905ec775'
    assert stream_source['membership_sha256'] == 'ef7ffec68a24f3ce30f2b709b1b8e09da90fa88ca530967ce44e0ba39b2d2d64'
    assert stream_source['reader_sha256'] == '0bf8d41c541109e11c89e3e7dbaf47b38234a87a995baf2e5f2f2e0b16d307f6'
    assert stream_source['identity_manifest_sha256'] == '6eb586b40bd68245e20205d896a5ae49700b72d61783b0e5ddc80a50e2242c87'
    assert vg_source['method'] == external_source['method'] == stream_source['method']
    stream_rows = [('openimages_test', 'test', *row) for row in stream_raw_rows]
    local_count = len(external_rows)
    external_rows, duplicates = merge_protected_rows(external_rows, stream_rows)
    merged_digest = hashlib.sha256(json.dumps(external_rows, separators=(',', ':')).encode()).hexdigest()
    assert vg_source['canonical_annotation_sha256'] == '56856b4386220f40d6b1b522a2ae199a8b4188765fd9ebb5321c1d0f4f2d7fbc'
    assert len(vg_rows) <= 89169 and len(external_rows) <= 137450
    manifest_raw = (folder / 'canonical_internal_external_identity_splits_v2.json').read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    assert manifest_sha == 'ade090fd4975e12e1ade2ac16a734f7b167f6630b38d1710cefa4176523f7b16'
    draft = {split: set(ids) for split, ids in json.loads(manifest_raw)['split_image_ids'].items()}
    assert not protected_internal('test', 'train')
    assert not protected_internal('val', 'train')
    assert not protected_internal('train', 'train')
    assert all(protected_internal(*pair) for pair in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    by_vg_hash, by_external_hash = defaultdict(list), defaultdict(list)
    for row in vg_rows:
        assert row[1] in ('train', 'val', 'test')
        for kind, digest in [('jpeg', row[2]), ('rgb', row[3])]:
            assert len(digest) == 64 and all(ch in '0123456789abcdef' for ch in digest)
            by_vg_hash[kind, digest].append(row)
    for row in external_rows:
        assert row[1] in ('val', 'test')
        for kind, digest in [('jpeg', row[3]), ('rgb', row[4])]:
            assert len(digest) == 64 and all(ch in '0123456789abcdef' for ch in digest)
            by_external_hash[kind, digest].append(row)
    internal_matches, external_matches = {}, {}
    for hash_key, members in sorted(by_vg_hash.items()):
        kind, digest = hash_key
        for source in members:
            if source[1] not in ('train', 'val'):
                continue
            for target in members:
                if protected_internal(source[1], target[1]):
                    assert source[4:6] == target[4:6], 'exact_pixel_shape_mismatch'
                    key = (source[0], target[0])
                    match = internal_matches.setdefault(key, {
                        'vg_id': source[0], 'canonical_split': source[1],
                        'protected_vg_id': target[0], 'protected_split': target[1],
                        'retained_in_current_draft': source[0] in draft[source[1]],
                        'matching_hashes': {}})
                    match['matching_hashes'][kind] = digest
            for target in by_external_hash.get(hash_key, []):
                assert source[4:6] == target[5:7], 'exact_pixel_shape_mismatch'
                key = (source[0], *target[:3])
                match = external_matches.setdefault(key, {
                    'vg_id': source[0], 'canonical_split': source[1],
                    'protected_dataset': target[0], 'protected_split': target[1],
                    'protected_image_id': target[2],
                    'retained_in_current_draft': source[0] in draft[source[1]],
                    'matching_hashes': {}})
                match['matching_hashes'][kind] = digest
    combined = list(internal_matches.values()) + list(external_matches.values())
    retained_ids = {split: sorted({row['vg_id'] for row in combined
                                  if row['canonical_split'] == split and row['retained_in_current_draft']})
                    for split in ('train', 'val')}
    evidence = {
        'schema_version': 2, 'method': 'Exact JPEG SHA256 or raw dimensions+RGB SHA256 equality; no perceptual threshold',
        'vg_snapshot_rows_sha256': vg_digest, 'external_snapshot_rows_sha256': external_digest,
        'vg_source': vg_source, 'external_source': external_source,
        'oi_stream_source':stream_source, 'oi_stream_snapshot_rows_sha256':stream_digest,
        'merged_protected_snapshot_rows_sha256':merged_digest,
        'local_stream_duplicate_identities':duplicates,
        'current_provisional_manifest_sha256': manifest_sha,
        'internal_matches': list(internal_matches.values()),
        'external_matches': list(external_matches.values()),
        'retained_draft_ids_with_exact_protected_matches': retained_ids,
    }
    raw = json.dumps(evidence, sort_keys=True, separators=(',', ':')).encode()
    assert len(raw) < 32 * 1024 ** 2, 'bounded_evidence_output_exceeded'
    evidence_sha = hashlib.sha256(raw).hexdigest()
    evidence_path = folder / ('exact_pixel_comparison_v2_' + evidence_sha + '.json')
    if evidence_path.exists():
        assert evidence_path.read_bytes() == raw
    else:
        with evidence_path.open('xb') as output:
            output.write(raw)
    report = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PARTIAL_EXACT_PIXEL_COMPARISON_COMPLETE',
        'script_sha256': hashlib.sha256((root / 'scripts/remote_caption_exact_pixel_compare_v2.py').read_bytes()).hexdigest(),
        'vg_indexed_images': len(vg_rows), 'vg_target_images': 89169,
        'external_indexed_images': len(external_rows), 'external_target_images': 137450,
        'local_external_indexed_images':local_count, 'oi_stream_indexed_images':len(stream_rows),
        'local_stream_duplicate_identities':duplicates,
        'oi_stream_snapshot_rows_sha256':stream_digest,
        'merged_protected_snapshot_rows_sha256':merged_digest,
        'oi_stream_phase_preregistration_sha256':stream_source['preregistration_sha256'],
        'vg_snapshot_rows_sha256': vg_digest, 'external_snapshot_rows_sha256': external_digest,
        'vg_snapshot_split_counts': dict(Counter(row[1] for row in vg_rows)),
        'internal_exact_match_pairs': len(internal_matches),
        'external_exact_match_pairs': len(external_matches),
        'retained_draft_ids_with_exact_protected_match_counts': {split: len(ids) for split, ids in retained_ids.items()},
        'retained_draft_ids_with_exact_protected_matches': retained_ids,
        'current_provisional_manifest_sha256': manifest_sha,
        'evidence_artifact_sha256': evidence_sha,
        'evidence_artifact_alias': '${REMOTE_HOME}/rsg-former/artifacts/caption_readiness/' + evidence_path.name,
        'wall_seconds': time.monotonic() - started, 'cpu_affinity_count': 1,
        'synthetic_split_direction_contract': 'PASS',
        'fingerprint_journals_modified': False, 'training_manifest_modified': False,
        'image_pixels_read_or_written': False, 'caption_reference_text_used': False,
        'benchmark_test_scored': False, 'live_v2_modified': False,
        'full_protected_coverage': False, 'perceptual_audit_complete': False,
        'clean_release_certified': False,
        'remaining': ['Complete VG and protected external pixel coverage',
                      'Independent perceptual candidate verification and exclusion policy',
                      'Versioned final exclusions before fresh training; absence of partial exact matches is not a clean certificate'],
    }
    (root / 'reports/CAPTION_EXACT_PIXEL_COMPARISON_V2.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
