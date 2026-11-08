"""Complete-unit native corpus metrics with bounded SPICE calls.

Caller supplies a private_metric_runtime in a dedicated evaluation process.
No dataset loading, model inference, selection or implicit missing-unit drops.
"""
import math
from collections.abc import Mapping


def validate_caption_units(references, predictions):
    if not isinstance(references, Mapping) or not isinstance(predictions, Mapping):
        raise ValueError('caption_inputs_must_be_mappings')
    if set(references) != set(predictions) or len(references) < 2:
        raise ValueError('caption_unit_sets_must_match_and_have_at_least_two_units')
    for identity, captions in references.items():
        if not isinstance(identity, str) or not identity:
            raise ValueError('caption_unit_id_must_be_nonempty_string')
        if not isinstance(captions, (list, tuple)) or not captions:
            raise ValueError('human_reference_list_empty_or_invalid')
        if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
            raise ValueError('human_reference_text_empty_or_invalid')
        if not isinstance(predictions[identity], str):
            raise ValueError('one_prediction_string_required_per_unit')
    return tuple(sorted(references))


def compute_caption_metrics(references, predictions, runtime, *, spice_batch_units=64):
    identities = validate_caption_units(references, predictions)
    if type(spice_batch_units) is not int or not 1 <= spice_batch_units <= 256:
        raise ValueError('invalid_bounded_spice_batch_units')
    # Native Java receives ordinal integer IDs; public unit IDs remain exact and
    # may include region composite keys. Sorting fixes native per-unit alignment.
    unit_ids = tuple(range(1, len(identities) + 1))
    raw_refs = {index: [{'caption': caption} for caption in references[identity]]
                for index, identity in zip(unit_ids, identities)}
    raw_preds = {index: [{'caption': predictions[identity]}]
                 for index, identity in zip(unit_ids, identities)}
    modules = runtime['modules']
    tokenizer = modules['PTBTokenizer'].PTBTokenizer()
    gts, res = tokenizer.tokenize(raw_refs), tokenizer.tokenize(raw_preds)
    if set(gts) != set(unit_ids) or set(res) != set(unit_ids):
        raise ValueError('tokenizer_changed_unit_membership')
    if any(len(gts[index]) != len(raw_refs[index]) or len(res[index]) != 1 for index in unit_ids):
        raise ValueError('tokenizer_changed_reference_or_prediction_counts')
    aggregate, per_unit = {}, {identity: {} for identity in identities}

    def save(name, score, scores):
        scalar = float(score)
        values = [float(value) for value in scores]
        if len(values) != len(identities) or not math.isfinite(scalar) or not all(map(math.isfinite, values)):
            raise ValueError('nonfinite_or_incomplete_metric:' + name)
        aggregate[name] = scalar
        for identity, value in zip(identities, values):
            per_unit[identity][name] = value

    bleu, bleu_units = modules['BLEU'].Bleu(4).compute_score(gts, res)
    if len(bleu) != 4 or len(bleu_units) != 4:
        raise ValueError('native_bleu_orders_contract_changed')
    for order in range(4):
        save('Bleu_' + str(order + 1), bleu[order], bleu_units[order])
    for name, scorer in [('CIDEr', modules['CIDEr'].Cider()), ('ROUGE_L', modules['ROUGE_L'].Rouge())]:
        # CIDEr document frequency is computed once on the entire reference
        # corpus. Corpus BLEU and METEOR likewise never become batch means.
        score, scores = scorer.compute_score(gts, res)
        save(name, score, scores)
    meteor = modules['METEOR'].Meteor()
    try:
        score, scores = meteor.compute_score(gts, res)
        save('METEOR', score, scores)
    finally:
        # The native destructor closes only this scorer's own child process.
        if meteor.lock.locked():
            meteor.lock.release()
        del meteor
    spice_values, batch_aggregates = [], []
    spice = modules['SPICE'].Spice()
    for start in range(0, len(unit_ids), spice_batch_units):
        batch = unit_ids[start:start + spice_batch_units]
        score, details = spice.compute_score({index: gts[index] for index in batch},
                                            {index: res[index] for index in batch})
        if len(details) != len(batch):
            raise ValueError('spice_changed_unit_count')
        values = [float(row['All']['f']) for row in details]
        if not math.isfinite(float(score)) or not all(map(math.isfinite, values)):
            raise ValueError('spice_all_f_undefined_no_silent_unit_drop')
        if not math.isclose(float(score), math.fsum(values) / len(values), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('native_spice_aggregate_contract_changed')
        spice_values.extend(values)
        batch_aggregates.append({'units': len(batch), 'native_aggregate': float(score)})
    save('SPICE', math.fsum(spice_values) / len(spice_values), spice_values)
    return {
        'aggregate': aggregate, 'per_unit': per_unit,
        'ordered_unit_ids': list(identities), 'evaluated_units': len(identities),
        'human_reference_counts': {identity: len(references[identity]) for identity in identities},
        'protocol': {'tokenization': 'native PTB once over complete corpus',
                     'CIDEr_document_frequency': 'native full reference corpus',
                     'BLEU_METEOR': 'native complete corpus aggregate, not sentence means',
                     'SPICE': 'native per-unit All.f; unit-weighted whole corpus mean',
                     'spice_batch_units': spice_batch_units, 'spice_batches': batch_aggregates,
                     'undefined_spice_all_f': 'fail; never silently drop expected units',
                     'score_scale': 'native pycocoevalcap values, no percentage conversion',
                     'ordinal_ids': 'sorted public unit IDs mapped to consecutive integers'},
    }
