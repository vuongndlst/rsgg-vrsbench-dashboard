"""Exact retrieval within a fixed63-bit Hamming radius; no pixel/score access."""
BANDS = ((0,13),(13,13),(26,13),(39,12),(51,12))
RADIUS = 4

def value(encoded):
    if not isinstance(encoded,str) or len(encoded)!=16 or any(c not in '0123456789abcdef' for c in encoded):
        raise ValueError('invalid_phash_encoding')
    result=int(encoded,16)
    if result>=1<<63:raise ValueError('not_63_bit_phash')
    return result

def bands(encoded):
    number=value(encoded)
    return tuple((index,(number>>offset)&((1<<width)-1))
                 for index,(offset,width) in enumerate(BANDS))

def distance(a,b):
    # Python3.9 compatibility for local fixtures.
    return bin(value(a)^value(b)).count('1')

def rotation_match(source,target):
    if len(source)!=4 or len(target)!=4:raise ValueError('four_rotations_required')
    candidates=[]
    for si,a in enumerate(source):
        for ti,b in enumerate(target):
            if set(bands(a)).isdisjoint(bands(b)):continue
            delta=distance(a,b)
            if delta<=RADIUS:candidates.append((delta,si*90,ti*90))
    return min(candidates) if candidates else None

def protected_direction(source_split,target_dataset,target_split):
    if target_dataset=='vg':
        return source_split=='train' and target_split in ('val','test') or source_split=='val' and target_split=='test'
    return source_split in ('train','val') and target_dataset in ('coco_karpathy','flickr30k_karpathy','openimages_test') and target_split in ('val','test')
