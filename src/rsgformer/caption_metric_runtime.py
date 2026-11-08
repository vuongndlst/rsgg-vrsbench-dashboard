"""Private pycocoevalcap launcher isolation for an audited Java8 subprocess.

Use only in a dedicated, single-threaded evaluation process. This changes
temporary locations and Java launcher flags, never metric algorithms or jars.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib
import json
import shutil

JAVA8_OPENS = frozenset('java.base/' + name + '=ALL-UNNAMED' for name in (
    'java.lang', 'java.util', 'java.util.concurrent', 'java.io',
    'java.math', 'java.net', 'java.text', 'java.lang.reflect'))
MODULES = {
    'PTBTokenizer': 'pycocoevalcap.tokenizer.ptbtokenizer',
    'METEOR': 'pycocoevalcap.meteor.meteor', 'SPICE': 'pycocoevalcap.spice.spice',
    'BLEU': 'pycocoevalcap.bleu.bleu', 'CIDEr': 'pycocoevalcap.cider.cider',
    'ROUGE_L': 'pycocoevalcap.rouge.rouge',
}
_active = False


def java8_spice_command(command):
    """Validate each invocation independently; reject unknown/duplicate flags."""
    if not command or command[0] != 'java' or 'spice-1.0.jar' not in command:
        raise ValueError('unexpected_spice_launcher')
    adjusted, removed, index = [], [], 0
    while index < len(command):
        if command[index] == '--add-opens':
            if index + 1 >= len(command) or command[index + 1] not in JAVA8_OPENS:
                raise ValueError('unexpected_java_module_flag')
            removed.append(command[index + 1])
            index += 2
        else:
            adjusted.append(command[index])
            index += 1
    if len(removed) != 8 or frozenset(removed) != JAVA8_OPENS:
        raise ValueError('spice_java_module_flag_contract_changed')
    return adjusted, removed


@contextmanager
def private_metric_runtime(project_root, scratch):
    global _active
    if _active:
        raise RuntimeError('nested_metric_runtime_forbidden')
    root, scratch = Path(project_root).resolve(), Path(scratch).resolve()
    if (root / 'artifacts/caption_readiness').resolve() not in scratch.parents:
        raise ValueError('metric_scratch_outside_audited_workspace')
    if not scratch.is_dir() or any(scratch.iterdir()):
        raise ValueError('metric_scratch_must_be_new_empty_directory')
    prepared = json.loads((root / 'reports/CAPTION_JAVA_PREPARATION.json').read_text())
    java = Path(prepared['java_path_alias'].replace('${REMOTE_HOME}', str(Path.home()))).resolve()
    if root not in java.parents or shutil.which('java') is None or Path(shutil.which('java')).resolve() != java:
        raise ValueError('private_java8_path_contract_failed')
    if hashlib.sha256(java.read_bytes()).hexdigest() != prepared['java_executable_sha256']:
        raise ValueError('private_java8_executable_hash_changed')
    if prepared['release_name'] != 'jdk8u504-b01':
        raise ValueError('java8_metric_release_contract_changed')
    preflight = json.loads((root / 'reports/CAPTION_RUNTIME_PREFLIGHT.json').read_text())
    modules, hashes = {}, {}
    for name, dotted in MODULES.items():
        module = importlib.import_module(dotted)
        source = Path(module.__file__)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashes[name] != preflight['scorers'][name]['source_sha256']:
            raise ValueError('scorer_source_hash_changed:' + name)
        for jar in preflight['scorers'][name]['jars']:
            if hashlib.sha256((source.parent / jar['relative_path']).read_bytes()).hexdigest() != jar['sha256']:
                raise ValueError('scorer_jar_hash_changed:' + name)
        modules[name] = module
    assets = importlib.import_module('pycocoevalcap.spice.get_stanford_models')
    spice, ptb = modules['SPICE'], modules['PTBTokenizer']
    originals = (assets.urlretrieve, spice.TEMP_DIR, spice.CACHE_DIR,
                 spice.subprocess.check_call, ptb.__file__)
    calls = []

    def forbidden_download(*args, **kwargs):
        raise RuntimeError('implicit_scorer_download_forbidden')

    def launch(command, *args, **kwargs):
        adjusted, removed = java8_spice_command(command)
        calls.append({'removed_add_opens_pairs': removed})
        return originals[3](adjusted, *args, **kwargs)

    ptb_dir = scratch / 'ptb'
    ptb_dir.mkdir()
    jar_name = ptb.STANFORD_CORENLP_3_4_1_JAR
    (ptb_dir / jar_name).symlink_to(Path(originals[4]).parent / jar_name)
    _active = True
    try:
        assets.urlretrieve = forbidden_download
        spice.TEMP_DIR, spice.CACHE_DIR = str(scratch / 'spice_tmp'), str(scratch / 'spice_cache')
        spice.subprocess.check_call = launch
        ptb.__file__ = str(ptb_dir / Path(originals[4]).name)
        yield {'modules': modules, 'source_hashes': hashes, 'spice_launcher_calls': calls,
               'metric_algorithm_modified': False, 'java_release': prepared['release_name']}
    finally:
        assets.urlretrieve = originals[0]
        spice.TEMP_DIR, spice.CACHE_DIR = originals[1:3]
        spice.subprocess.check_call = originals[3]
        ptb.__file__ = originals[4]
        _active = False
