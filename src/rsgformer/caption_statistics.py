"""Paired image-cluster bootstrap for fixed-corpus per-unit caption scores.

CIDEr document frequencies stay those of the evaluated corpus. This estimates
uncertainty conditional on those frequencies; it does not recompute CIDEr for
each bootstrap corpus. Global corpus BLEU/METEOR cannot use this mean estimator.
"""
import math
import numpy as np


def paired_image_bootstrap(left, right, parent_images, *, repetitions, seed):
    """Compare matched units; resample parent images, retaining all their units.

    COCO/Flickr/OI: one unit per image. VG region evaluations: several units can
    share an image, so independent region resampling would understate dependence.
    The point estimate remains the mean over units, including any zero penalties
    already assigned by a separately locked eligibility/coverage protocol.
    """
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1000:
        raise ValueError('bootstrap_repetitions_must_be_integer_at_least_1000')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('bootstrap_seed_must_be_nonnegative_integer')
    if not left or set(left) != set(right) or set(left) != set(parent_images):
        raise ValueError('paired_unit_membership_mismatch_or_empty')
    if not all(isinstance(key, str) and key for key in left):
        raise ValueError('unit_ids_must_be_nonempty_strings')
    if not all(isinstance(value, str) and value for value in parent_images.values()):
        raise ValueError('parent_image_ids_must_be_nonempty_strings')
    units = sorted(left)
    a = np.asarray([left[key] for key in units], dtype=np.float64)
    b = np.asarray([right[key] for key in units], dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired_scores_must_be_finite_scalars')
    images = sorted(set(parent_images.values()))
    if len(images) < 2:
        raise ValueError('at_least_two_parent_images_required')
    image_index = {identity: index for index, identity in enumerate(images)}
    group = np.asarray([image_index[parent_images[key]] for key in units], dtype=np.int64)
    counts = np.bincount(group, minlength=len(images))
    deltas = np.bincount(group, weights=b - a, minlength=len(images))
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    # One draw at a time: O(images + units + repetitions) working memory.
    for index in range(repetitions):
        draw = rng.integers(0, len(images), size=len(images))
        estimates[index] = deltas[draw].sum() / counts[draw].sum()
    lower, upper = np.quantile(estimates, [.025, .975], method='linear')
    point = float((b - a).mean())
    assert math.isfinite(point) and np.isfinite(estimates).all()
    return {'estimand': 'right_minus_left_mean_over_matched_units',
            'left_mean': float(a.mean()), 'right_mean': float(b.mean()),
            'paired_delta': point, 'ci95_percentile': [float(lower), float(upper)],
            'repetitions': repetitions, 'seed': seed, 'unit_count': len(units),
            'parent_image_count': len(images),
            'resampling_unit': 'parent_image_retaining_all_caption_units',
            'units_per_image_min': int(counts.min()), 'units_per_image_max': int(counts.max()),
            'unit_membership_matched': True, 'corpus_document_frequencies_recomputed': False,
            'conditional_on_fixed_corpus_per_unit_scores': True}
