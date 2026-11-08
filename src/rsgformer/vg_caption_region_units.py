"""Geometry-only model inputs and separate human region evaluation references.

Provided crops are conditional inputs, not learned region detections. The caller
must supply raw decoded/audited image dimensions; this module does not load data,
images, models or scores, and does not choose a workload or evaluation subset.
"""
import math


def positive_identity(value):
    if type(value) is not int or value <= 0:
        raise ValueError('invalid_region_or_image_identity')
    return value


def prepare_provided_crop_units(image_id, image_size, regions):
    positive_identity(image_id)
    if len(image_size) != 2 or any(type(value) is not int or value <= 0 for value in image_size):
        raise ValueError('invalid_audited_image_dimensions')
    width, height = image_size
    inputs, references, rejected = {}, {}, []
    seen = set()
    for region in regions:
        identity = positive_identity(region['region_id'])
        if positive_identity(region['image_id']) != image_id:
            raise ValueError('region_parent_identity_mismatch')
        if identity in seen:
            raise ValueError('duplicate_region_identity')
        seen.add(identity)
        unit = 'vg:' + str(image_id) + ':region:' + str(identity)
        values = [region[key] for key in ('x', 'y', 'width', 'height')]
        reason = None
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            reason = 'nonfinite_or_invalid_geometry'
        else:
            x, y, w, h = values
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                reason = 'nonpositive_or_negative_geometry'
            elif not math.isfinite(x+w) or not math.isfinite(y+h) or x+w > width or y+h > height:
                reason = 'original_box_not_fully_contained'
        phrase = region.get('phrase')
        if reason is None and (not isinstance(phrase, str) or not phrase.strip()):
            reason = 'empty_or_invalid_human_reference'
        if reason:
            rejected.append({'unit_id': unit, 'parent_image_id': image_id, 'reason': reason})
            continue
        crop = [math.floor(x), math.floor(y), math.ceil(x+w), math.ceil(y+h)]
        assert 0 <= crop[0] < crop[2] <= width and 0 <= crop[1] < crop[3] <= height
        inputs[unit] = {'unit_id': unit, 'parent_image_id': image_id, 'region_id': identity,
                        'image_size': [width, height], 'crop_xyxy': crop,
                        'scope': 'provided_crop_conditional_caption'}
        references[unit] = {'parent_image_id': image_id, 'human_references': [phrase]}
    assert len(inputs) + len(rejected) == len(seen)
    assert set(inputs) == set(references)
    return ({key: inputs[key] for key in sorted(inputs)},
            {key: references[key] for key in sorted(references)}, rejected)
