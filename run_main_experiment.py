"""Execute unchanged sealed experiment settings with honest upload provenance.

The original runners require a verified local Moodle receipt. This separate
launcher implements the user's later explicit request to execute now while
keeping that missing evidence UNVERIFIED. It does not create a Moodle receipt,
claim protocol submission, or change the sealed inputs or numerical code.
Use this launcher for reproduction/resume of this explicitly recorded run.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent
RECORD = ROOT / 'protocol/freeze_manifest.json'
USER_REQUEST = 'давай уже запустим основной эксперимент на 2000 вопросов для поиска и 1000 для генерации, а потом сделаем отчёт по реальным результатам'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
        temporary = f.name
    os.replace(temporary, path)


def verify_local(root, cfg, paths, source_names):
    import freeze_gate as gate
    root = Path(root).resolve()
    if root != ROOT:
        raise ValueError('Unexpected experiment root')
    seal = read(root / gate.SEAL_PATH)
    _, sources, environment = gate._verify_plan(root, cfg, paths, source_names, seal)
    return seal, sources, environment


def execution_proof(root, cfg, paths, source_names):
    seal, sources, environment = verify_local(root, cfg, paths, source_names)
    record = read(RECORD)
    expected = {
        'status': 'execution_requested_upload_unverified',
        'moodle_upload_verified': False,
        'moodle_receipt_available': False,
        'upload_chronology_verified': False,
        'sealed_plan_sha256': sha(ROOT / 'protocol/sealed_plan.json'),
        'protocol_sha256': seal['protocol']['sha256'],
        'execution_launcher_sha256': sha(Path(__file__)),
        'user_request': USER_REQUEST,
    }
    if not all(record.get(k) == v for k, v in expected.items()):
        raise ValueError('Execution provenance is missing, altered or mislabeled')
    return {
        **expected,
        'evidence_basis': 'unchanged_local_seal_and_explicit_user_execution_request',
        'freeze_manifest_sha256': sha(RECORD),
        'config_sha256': seal['inputs']['config']['sha256'],
        'preflight_report_sha256': seal['preflight']['report']['sha256'],
        'input_sha256': {n: v['sha256'] for n, v in seal['inputs'].items()},
        'source_sha256': sources,
        'package_versions': environment['package_versions'],
        'python_version': environment['python_version'],
        'nltk_english_resource_sha256': environment['nltk_english_resource_sha256'],
        'reported_upload_time_unverified': '2026-11-08T10:00:00+05:00',
        'verification_scope': 'Sealed inputs/code/environment verified locally; Moodle submission and chronology not verified',
        'deviation': record['deviation'],
    }


def initialise_record():
    import freeze_gate as gate
    cfg = read(ROOT / 'experiment_config.json')
    paths = gate.expected_input_paths(ROOT, cfg)
    seal, _, _ = verify_local(ROOT, cfg, paths, gate.REQUIRED_SOURCES)
    if RECORD.exists():
        execution_proof(ROOT, cfg, paths, gate.REQUIRED_SOURCES)
        return
    record = {
        'schema_version': 1,
        'status': 'execution_requested_upload_unverified',
        'recorded_at_utc': utc(),
        'user_request': USER_REQUEST,
        'moodle_upload_verified': False,
        'moodle_receipt_available': False,
        'upload_chronology_verified': False,
        'reported_upload_time_unverified': '2026-11-08T10:00:00+05:00',
        'chronology_note': 'User-reported upload time predates local seal at 2026-11-08T10:56:03+05:00; receipt and uploaded bytes have not been checked.',
        'sealed_plan_sha256': sha(ROOT / gate.SEAL_PATH),
        'protocol_sha256': seal['protocol']['sha256'],
        'execution_launcher_sha256': sha(Path(__file__)),
        'deviation': 'Execution proceeded at the user explicit request without verified Moodle upload evidence. Original sealed data, models, prompts, chunking, metrics and analysis plan are unchanged. This record does not establish compliance with the assignment prior-upload requirement.',
    }
    # Exclusive creation: never replace an existing upload or execution record.
    with RECORD.open('x', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    execution_proof(ROOT, cfg, paths, gate.REQUIRED_SOURCES)


def invoke(module, arguments):
    previous = sys.argv
    try:
        sys.argv = [str(ROOT / (module.__name__ + '.py')), *arguments]
        module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--execute-with-unverified-upload', action='store_true')
    parser.add_argument('--resume-generation', type=Path)
    args = parser.parse_args()
    import freeze_gate as gate
    cfg = read(ROOT / 'experiment_config.json')
    paths = gate.expected_input_paths(ROOT, cfg)
    verify_local(ROOT, cfg, paths, gate.REQUIRED_SOURCES)
    if args.check_only:
        from check_final_inputs import validate_inputs
        corpus, questions, titles, annotations = validate_inputs(cfg, paths)
        print(json.dumps({'status': 'local_seal_and_final_inputs_verified',
                          'articles': len(titles), 'retrieval_questions': len(questions),
                          'generation_questions': sum(q['for_generation'] for q in questions),
                          'gold_annotations': annotations, 'moodle_upload_verified': False,
                          'models_loaded': False, 'execution_requested': False}, indent=2))
        return
    if not args.execute_with_unverified_upload:
        parser.error('Explicit --execute-with-unverified-upload is required for this separately disclosed execution route')
    final = ROOT / 'final'
    final.mkdir(exist_ok=True)
    with (final / 'main_experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialise_record()
        # Inject only the provenance validator; original inference and scoring
        # functions and all sealed files stay byte-for-byte unchanged.
        gate.load_freeze = execution_proof
        provenance = final / 'execution_provenance'
        provenance.mkdir(exist_ok=True)
        for source in [Path(__file__), RECORD, ROOT/'protocol/sealed_plan.json', ROOT/'protocol/Protocol_Freeze.docx', ROOT/'analysis_plan.json']:
            target = provenance / source.name
            if target.exists() and sha(target) != sha(source):
                raise ValueError('Execution provenance snapshot changed: ' + target.name)
            if not target.exists():shutil.copyfile(source, target)
        state_path = final / 'pipeline_manifest.json'
        state = read(state_path) if state_path.exists() else {
            'purpose':'full_final_experiment', 'status':'created',
            'created_at_utc':utc(), 'moodle_upload_verified':False,
            'expected_retrieval_records':6000, 'expected_generated_answers':3000,
            'execution_launcher_sha256':sha(Path(__file__)), 'attempts':[]}
        if state['execution_launcher_sha256'] != sha(Path(__file__)):
            raise ValueError('Launcher changed since pipeline creation')
        attempt = {'started_at_utc':utc(), 'status':'running'}
        state['attempts'].append(attempt)
        try:
            import final_retrieval as retrieval
            import final_generation as generation
            if args.resume_generation:
                out=(ROOT/args.resume_generation).resolve()
                manifest=read(out/'run_manifest.json')
                source=ROOT/manifest['identity']['retrieval_source']
                state.update(status='generation_running',retrieval_run=str(source.relative_to(ROOT)),generation_run=str(out.relative_to(ROOT)))
                write(state_path,state)
                invoke(generation,['--resume',str(out)])
            else:
                existing = [p for p in (final/'retrieval_runs').glob('*/run_manifest.json')
                            if read(p).get('status')=='completed' and read(p).get('freeze_manifest_sha256')==sha(RECORD)]
                if len(existing)>1:raise ValueError('Multiple completed retrieval runs; choose a run explicitly')
                if existing:
                    source=existing[0].parent
                else:
                    before=set((final/'retrieval_runs').glob('*'))
                    state.update(status='retrieval_running');write(state_path,state)
                    invoke(retrieval,['--execute'])
                    created=set((final/'retrieval_runs').glob('*'))-before
                    if len(created)!=1:raise ValueError('Cannot identify the single new retrieval run')
                    source=created.pop()
                state.update(status='generation_running',retrieval_run=str(source.relative_to(ROOT)))
                write(state_path,state)
                gc.collect()
                before=set((final/'generation_runs').glob('*'))
                invoke(generation,['--retrieval-run',str(source)])
                created=set((final/'generation_runs').glob('*'))-before
                if len(created)!=1:raise ValueError('Cannot identify the single new generation run')
                state['generation_run']=str(created.pop().relative_to(ROOT))
            state.update(status='completed',completed_at_utc=utc())
            attempt.update(status='completed',completed_at_utc=utc())
            write(state_path,state)
            print('MAIN EXPERIMENT COMPLETED',json.dumps(state,ensure_ascii=False),flush=True)
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',stopped_at_utc=utc())
            attempt.update(status=state['status'],error=f'{type(exc).__name__}: {exc}',stopped_at_utc=utc())
            write(state_path,state)
            raise


if __name__=='__main__':
    main()
