"""Model-free, frozen-plan analysis. Only complete paired runs are accepted.
Default final mode requires 20 articles/2000 retrieval/1000 generation questions.
--development is explicit, descriptive only, and makes no hypothesis decisions.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import random
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

METHODS = ['fixed', 'overlap', 'semantic']
PAIRS = [('overlap', 'fixed'), ('semantic', 'fixed'), ('semantic', 'overlap')]
NAMES = ['omnibus', 'overlap_minus_fixed', 'semantic_minus_fixed', 'semantic_minus_overlap']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def norm(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def scores(pred, refs):
    require(bool(refs), 'Gold references are empty')
    p = norm(pred).split()
    f1 = []
    for ref in refs:
        g = norm(ref).split()
        n = sum((Counter(p) & Counter(g)).values())
        f1.append(2 * n / (len(p) + len(g)) if n else 0.0)
    return dict(exact_match=float(any(norm(pred) == norm(g) for g in refs)), f1=max(f1))


def hit(chunks, article, spans):
    require(bool(spans), 'Gold spans are empty')
    return float(any(c['article'] == article and c['start'] <= s < e <= c['end'] for c in chunks for s, e in spans))


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, previous = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        previous = min(1.0, max(previous, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def primary_tests(matrix, plan, draws=None):
    matrix = np.asarray(matrix, dtype=float)
    require(matrix.ndim == 2 and matrix.shape[1] == 3 and np.isfinite(matrix).all(), 'Invalid article primary matrix')
    count, draws = len(matrix), draws or plan['monte_carlo_draws']
    tol = plan['numerical_comparison_tolerance']
    observed = count * np.sum((matrix.mean(axis=0) - matrix.mean()) ** 2)
    permutations = np.array(list(itertools.permutations(range(3))))
    rng, extreme = np.random.default_rng(plan['seed']), 0
    for first in range(0, draws, 1000):
        size = min(1000, draws - first)
        indices = permutations[rng.integers(0, 6, size=(size, count))]
        simulated = np.take_along_axis(np.broadcast_to(matrix, (size, count, 3)), indices, axis=2).mean(axis=1)
        stats = count * np.sum((simulated - matrix.mean()) ** 2, axis=1)
        extreme += int(np.sum(stats >= observed - tol))
    pvalues, statistics = [(extreme + 1) / (draws + 1)], [float(observed)]
    differences = []
    for left, right in PAIRS:
        d = matrix[:, METHODS.index(left)] - matrix[:, METHODS.index(right)]
        differences.append(float(d.mean()))
        observed_pair, extreme = abs(float(d.mean())), 0
        rng = np.random.default_rng(plan['seed'])
        for first in range(0, draws, 1000):
            size = min(1000, draws - first)
            signs = rng.integers(0, 2, size=(size, count)) * 2 - 1
            simulated = np.abs(np.sum(signs * d, axis=1) / count)
            extreme += int(np.sum(simulated >= observed_pair - tol))
        pvalues.append((extreme + 1) / (draws + 1))
        statistics.append(observed_pair)
    adjusted = holm(pvalues)
    return dict(raw_p=dict(zip(NAMES, pvalues)), adjusted_p=dict(zip(NAMES, adjusted)),
                statistics=dict(zip(NAMES, statistics)), draws=draws, seed=plan['seed'],
                paired_differences=dict(zip(NAMES[1:], differences)), omnibus_gate=adjusted[0] < plan['alpha'],
                h1_supported=adjusted[0] < plan['alpha'],
                h2_supported=adjusted[0] < plan['alpha'] and adjusted[1] < plan['alpha'] and differences[0] > 0,
                practical_flag_abs_difference_pp=2.0,
                pairwise_claims={n: adjusted[0] < plan['alpha'] and adjusted[i] < plan['alpha'] for i, n in enumerate(NAMES[1:], 1)},
                assumptions=plan['assumptions'])


def pooled_contrast(aggregates, weights=None):
    a = np.asarray(aggregates, dtype=float)
    total = a.sum(axis=0) if weights is None else np.sum(a * np.asarray(weights)[:, None], axis=0)
    return float(total[0] / total[1] - total[2] / total[3]) if total[1] > 0 and total[3] > 0 else None


def conditional(rows, articles, indicator, metric, plan):
    aggregates = np.zeros((len(articles), 4))
    index = {a: i for i, a in enumerate(articles)}
    for row in rows:
        state = row['retrieval_metrics']['hit_at_3'] if indicator == 'raw' else row['visible_evidence_hit']
        column = 0 if state else 2
        aggregates[index[row['article']], column] += row['answer_metrics'][metric]
        aggregates[index[row['article']], column + 1] += 1
    rng = np.random.default_rng(plan['seed'])
    indices = rng.integers(0, len(articles), size=(plan['resamples'], len(articles)))
    totals = aggregates[indices].sum(axis=1)
    valid = (totals[:, 1] > 0) & (totals[:, 3] > 0)
    values = totals[valid, 0] / totals[valid, 1] - totals[valid, 2] / totals[valid, 3]
    ci = [float(v) for v in np.percentile(values, plan['confidence_percentiles'])] if len(values) else None
    total = aggregates.sum(axis=0)
    return dict(contrast=pooled_contrast(aggregates), hit_records=int(total[1]), miss_records=int(total[3]),
                mean_hit=float(total[0]/total[1]) if total[1] else None,
                mean_miss=float(total[2]/total[3]) if total[3] else None,
                hit_articles=int(np.sum(aggregates[:, 1] > 0)), miss_articles=int(np.sum(aggregates[:, 3] > 0)),
                defined_draws=int(valid.sum()), bootstrap_draws=plan['resamples'], ci95=ci)


def h3_analysis(rows, article_order, plan):
    p = plan['H3_exploratory']
    pooled = conditional(rows, article_order, 'raw', 'f1', p)
    within = {s: conditional([r for r in rows if r['strategy'] == s], article_order, 'raw', 'f1', p) for s in METHODS}
    adequate = (pooled['hit_articles'] >= p['minimum_articles_contributing_to_each_hit_state']
                and pooled['miss_articles'] >= p['minimum_articles_contributing_to_each_hit_state']
                and pooled['defined_draws'] >= p['minimum_defined_bootstrap_draws'])
    positive = (pooled['contrast'] is not None and pooled['contrast'] > 0 and pooled['ci95'] is not None
                and pooled['ci95'][0] > 0 and all(v['contrast'] is not None and v['contrast'] > 0 for v in within.values()))
    return dict(raw_hit_f1=pooled, within_method=within, adequacy_passed=adequate,
                status='consistent_exploratory' if adequate and positive else 'inconclusive_or_not_supported',
                sensitivity_visible_f1=conditional(rows, article_order, 'visible', 'f1', p),
                sensitivity_raw_hit_em=conditional(rows, article_order, 'raw', 'exact_match', p), limitations=p['limitations'])


def verify_run(path, purpose):
    manifest = read(path / 'run_manifest.json')
    require(manifest['purpose'] == purpose and manifest['status'] == 'completed', 'Run is incomplete or wrong mode: ' + str(path))
    require(bool(manifest.get('output_sha256')), 'No output hashes')
    for name, expected in manifest['output_sha256'].items():
        target = (path / name).resolve()
        require(path.resolve() in target.parents and sha(target) == expected, 'Artifact hash mismatch: ' + name)
    identity = manifest.get('identity', manifest)
    for name, expected in identity.get('source_sha256', {}).items():
        require(sha(path / 'source_snapshot' / name) == expected, 'Source snapshot mismatch: ' + name)
    return manifest


def load_runs(retrieval, generation, plan_path, development, allow_unverified_upload=False):
    plan, plan_hash = read(plan_path), sha(plan_path)
    prefix = 'development' if development else 'final'
    rm = verify_run(retrieval, prefix + '_retrieval')
    gm = verify_run(generation, prefix + '_generation')
    require(sha(retrieval/'config_snapshot.json') == sha(generation/'config_snapshot.json'), 'Config snapshots differ')
    require(sha(retrieval/'run_manifest.json') == gm.get('retrieval_manifest_sha256', gm.get('identity', {}).get('retrieval_manifest_sha256')), 'Generation links another retrieval run')
    for name in ['corpus.json', 'questions.json', 'selection_manifest.json']:
        require(sha(retrieval/name) == sha(generation/name), 'Input snapshots differ: ' + name)
    cfg, corpus = read(retrieval/'config_snapshot.json'), read(retrieval/'corpus.json')
    questions, selection = read(retrieval/'questions.json'), read(retrieval/'selection_manifest.json')
    articles, ids = selection['article_titles'], selection['retrieval_question_ids']
    gids = selection['generation_question_ids']
    expected = (3, 9, 9, 3, 3) if development else (20, 2000, 1000, 100, 50)
    require((len(articles), len(ids), len(gids)) == expected[:3] and len(set(ids)) == len(ids)
            and len(set(gids)) == len(gids) and len(set(articles)) == len(articles), 'Sample sizes/uniqueness differ')
    require(list(corpus) == articles and [q['id'] for q in questions] == ids and set(gids) <= set(ids), 'Input IDs/order mismatch')
    require([q['id'] for q in questions if q['for_generation']] == gids, 'Generation flags/order mismatch')
    require(Counter(q['article'] for q in questions) == Counter({a: expected[3] for a in articles}), 'Retrieval article imbalance')
    require(Counter(q['article'] for q in questions if q['for_generation']) == Counter({a: expected[4] for a in articles}), 'Generation article imbalance')
    require(rm['question_ids'] == ids and gm['question_ids'] == gids, 'Run manifest ID mismatch')
    require(rm['completed_records'] == len(ids)*3 if 'completed_records' in rm else development, 'Retrieval completion count mismatch')
    require(gm['completed_records'] == len(gids)*3, 'Generation completion count mismatch')
    if not development:
        require(cfg['status'] == 'frozen' and gm.get('final_run') is True, 'Final data must be frozen')
        require(plan['evaluation_design']['article_order'] == articles, 'Frozen plan article order differs')
        proof = rm['freeze_proof']
        require(proof['source_sha256']['analysis_plan.json'] == plan_hash, 'Frozen analysis plan hash differs')
        require(proof.get('status') == 'uploaded' or allow_unverified_upload,
                'Upload provenance is unverified; explicit --allow-unverified-upload is required to disclose this deviation')
        require(gm['identity']['freeze_proof'] == proof, 'Freeze proofs differ')
        if proof.get('status') != 'uploaded':
            require(proof.get('moodle_upload_verified') is False and proof.get('upload_chronology_verified') is False,
                    'Unverified execution must explicitly disclose missing upload/chronology evidence')
            snapshot = retrieval.parent.parent / 'execution_provenance'
            for filename, key in [('run_main_experiment.py','execution_launcher_sha256'),('freeze_manifest.json','freeze_manifest_sha256'),('sealed_plan.json','sealed_plan_sha256'),('Protocol_Freeze.docx','protocol_sha256')]:
                require(sha(snapshot/filename) == proof[key], 'Execution provenance snapshot hash differs: '+filename)
            require(sha(snapshot/'analysis_plan.json') == plan_hash,'Execution analysis plan snapshot differs')
        for name, expected_hash in plan['evaluation_design']['data_sha256'].items():
            actual = rm['input_sha256']['raw_dataset'] if name == 'dev-v1.1.json' else sha(retrieval/name)
            require(actual == expected_hash, 'Frozen plan data hash differs: ' + name)
    retrieval_rows, generation_rows, chunks_by_method = [], [], {}
    qmap = {q['id']: q for q in questions}
    for method in METHODS:
        chunks = read(retrieval/method/'chunks.json'); chunks_by_method[method] = chunks
        by_id, by_article = {c['chunk_id']: c for c in chunks}, defaultdict(list)
        require(len(by_id) == len(chunks), 'Duplicate chunks')
        for c in chunks:
            require(c['article'] in corpus and 0 <= c['start'] < c['end'] <= len(corpus[c['article']])
                    and corpus[c['article']][c['start']:c['end']] == c['text'], 'Chunk source mismatch')
            by_article[c['article']].append(c)
        rr, gr = read(retrieval/method/'per_question.json'), read(generation/method/'per_question.json')
        require([r['id'] for r in rr] == ids and [r['id'] for r in gr] == gids, 'Paired method IDs/order mismatch')
        lookup = {}
        for r in rr:
            require(all(r[k] == v for k, v in qmap[r['id']].items()), 'Question content changed')
            for answer, (start, end) in zip(r['answers'], r['answer_spans']):
                require(corpus[r['article']][start:end] == answer, 'Gold source span mismatch')
            ranked = r['retrieved_results']
            require(len(ranked) == 5 and [x['rank'] for x in ranked] == [1,2,3,4,5], 'Expected top5 ranks')
            require(len({x['chunk']['chunk_id'] for x in ranked}) == 5, 'Repeated top5 chunks')
            for x in ranked:
                require(x['chunk'] == by_id[x['chunk']['chunk_id']] and math.isfinite(x['score']), 'Bad retrieval chunk/score')
            checked = {f'hit_at_{k}': hit([x['chunk'] for x in ranked[:k]], r['article'], r['answer_spans']) for k in [1,3,5]}
            require(checked == r['retrieval_metrics'], 'Stored retrieval scores differ')
            coverage = hit(by_article[r['article']], r['article'], r['answer_spans'])
            require(checked['hit_at_1'] <= checked['hit_at_3'] <= checked['hit_at_5'] <= coverage, 'Coverage hierarchy failed')
            item = dict(r, strategy=method, corpus_answer_span_coverage=coverage)
            retrieval_rows.append(item); lookup[r['id']] = item
        for r in gr:
            require(all(r[k] == v for k, v in lookup[r['id']].items() if k not in ['strategy','corpus_answer_span_coverage']), 'Generation source differs')
            actual = scores(r['prediction'], r['answers'])
            require(all(math.isclose(actual[k], r['answer_metrics'][k], abs_tol=1e-12) for k in actual), 'Stored EM/F1 differ')
            packed = r['packed_prompt']
            require(packed['prompt'] == cfg['generation']['prompt_template'].format(question=r['question'], context=packed['context']), 'Prompt differs')
            require(len(r['input_token_ids']) == len(r['attention_mask']) == packed['input_tokens'] <= 512, 'Prompt token counts differ')
            for c in packed['visible_chunks']:
                require(corpus[c['article']][c['start']:c['end']] == c['text'], 'Visible source mismatch')
            visible = hit(packed['visible_chunks'], r['article'], r['answer_spans'])
            require(visible == r['visible_evidence_hit'] <= r['retrieval_metrics']['hit_at_3'], 'Visible coverage differs')
            generation_rows.append(dict(r, answer_metrics=actual))
    jsonl = [json.loads(s) for s in (generation/'per_question.jsonl').read_text().splitlines() if s.strip()]
    require(jsonl == [r for method in METHODS for r in read(generation/method/'per_question.json')], 'JSONL/per-method arrays differ')
    upload_status = rm.get('freeze_proof', {}).get('status') if not development else 'not_applicable_development'
    provenance = dict(mode=prefix, plan_path=str(plan_path), plan_sha256=plan_hash,
                      upload_verification=('RECORDED_LOCAL_UPLOAD_ATTESTATION' if upload_status == 'uploaded' else 'UNVERIFIED' if not development else 'NOT_APPLICABLE'),
                      protocol_chronology=('See recorded upload/seal timestamps; Moodle server not authenticated' if upload_status == 'uploaded' else 'UNRESOLVED: no verified Moodle upload evidence' if not development else 'Development only'),
                      retrieval_path=str(retrieval), generation_path=str(generation),
                      retrieval_manifest_sha256=sha(retrieval/'run_manifest.json'), generation_manifest_sha256=sha(generation/'run_manifest.json'),
                      retrieval_manifest=rm, generation_manifest=gm)
    return plan, articles, retrieval_rows, generation_rows, chunks_by_method, provenance


def csv_write(path, rows):
    if rows:
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize_tables(retrieval, generation, articles, plan):
    rng = np.random.default_rng(plan['confidence_intervals']['seed'])
    draws = rng.integers(0, len(articles), size=(plan['confidence_intervals']['resamples'], len(articles)))
    tables, article_rows, matrices = {}, [], {}
    for phase, rows, keys in [('retrieval', retrieval, ['hit_at_1','hit_at_3','hit_at_5','corpus_answer_span_coverage']),
                              ('generation', generation, ['exact_match','f1','hit_at_3','visible_evidence_hit'])]:
        table=[]
        for method in METHODS:
            subset=[r for r in rows if r['strategy']==method]
            result=dict(strategy=method,n_questions=len(subset))
            for key in keys:
                def get(r):
                    if key in ['exact_match','f1']: return r['answer_metrics'][key]
                    if key.startswith('hit_at'): return r['retrieval_metrics'][key]
                    return r[key]
                values=np.array([np.mean([get(r) for r in subset if r['article']==a]) for a in articles])
                matrices[(phase,method,key)]=values
                boot=values[draws].mean(axis=1); ci=np.percentile(boot,[2.5,97.5])
                result[key]=float(values.mean());result[key+'_sum']=float(sum(get(r) for r in subset))
                result[key+'_ci_low']=float(ci[0]);result[key+'_ci_high']=float(ci[1])
            if phase=='generation':result['context_truncated_count']=sum(r['packed_prompt']['context_truncated'] for r in subset)
            table.append(result)
        tables[phase]=table
        for article in articles:
            for method in METHODS:
                subset=[r for r in rows if r['article']==article and r['strategy']==method]
                result=dict(phase=phase,article=article,strategy=method,n_questions=len(subset))
                for key in keys:
                    result[key]=float(np.mean([r['answer_metrics'][key] if key in ['exact_match','f1'] else r['retrieval_metrics'][key] if key.startswith('hit_at') else r[key] for r in subset]))
                article_rows.append(result)
    differences=[]
    for phase,keys in [('retrieval',['hit_at_1','hit_at_3','hit_at_5','corpus_answer_span_coverage']),('generation',['exact_match','f1','hit_at_3','visible_evidence_hit'])]:
        for key in keys:
            for left,right in PAIRS:
                d=matrices[(phase,left,key)]-matrices[(phase,right,key)]
                ci=np.percentile(d[draws].mean(axis=1),[2.5,97.5])
                differences.append(dict(phase=phase,metric=key,comparison=left+'_minus_'+right,difference=float(d.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
    return tables,article_rows,differences,np.column_stack([matrices[('retrieval',s,'hit_at_3')] for s in METHODS])


def draw_figure(path, table, metrics, title, mode, clusters):
    from PIL import Image,ImageDraw,ImageFont
    image=Image.new('RGB',(1600,1000),'white');d=ImageDraw.Draw(image)
    font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
    font=lambda size:ImageFont.truetype(font_path,size) if Path(font_path).exists() else ImageFont.load_default()
    d.text((100,45),title,fill='#14243a',font=font(42))
    d.text((100,105),f'{mode.upper()} | {table[0]["n_questions"]} paired questions per method | {clusters} article clusters',fill='#3b4b60',font=font(25))
    x0,y0,x1,y1=130,200,1500,800
    for rate in range(0,101,20):
        y=y1-(y1-y0)*rate/100;d.line((x0,y,x1,y),fill='#dde3e9',width=2);d.text((65,y-15),str(rate),fill='black',font=font(24))
    d.line((x0,y0,x0,y1,x1,y1),fill='black',width=3);d.text((130,155),'Rate (%)',fill='black',font=font(24))
    colors=['#245f9e','#cb7939','#4d927b'];group=(x1-x0)/3;bar=75;gap=18
    for i,row in enumerate(table):
        center=x0+group*(i+.5)
        for j,key in enumerate(metrics):
            x=center+(j-(len(metrics)-1)/2)*(bar+gap);value=row[key]*100;y=y1-(y1-y0)*value/100
            d.rectangle((x-bar/2,y,x+bar/2,y1),fill=colors[j])
            lo=y1-(y1-y0)*row[key+'_ci_low'];hi=y1-(y1-y0)*row[key+'_ci_high']
            d.line((x,lo,x,hi),fill='black',width=3)
            for yy in [lo,hi]:d.line((x-12,yy,x+12,yy),fill='black',width=3)
            d.text((x-35,max(y0-25,min(y,hi)-35)),f'{value:.1f}',fill='black',font=font(23))
        label={'fixed':'Fixed 100','overlap':'Overlap 100/20','semantic':'Semantic p90 / cap100'}[row['strategy']]
        d.text((center-145,825),label,fill='black',font=font(26))
    for j,key in enumerate(metrics):
        x=130+j*380;d.rectangle((x,885,x+25,910),fill=colors[j]);d.text((x+40,882),key.replace('_',' '),fill='black',font=font(25))
    d.text((100,945),'Marginal 95% whole-article percentile intervals; 2000 resamples, seed42. Fixed corpus/models.',fill='#3b4b60',font=font(23))
    image.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval-run',type=Path,required=True);parser.add_argument('--generation-run',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=Path(__file__).with_name('analysis_plan.json'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--development',action='store_true')
    parser.add_argument('--allow-unverified-upload',action='store_true',help='Disclose missing Moodle evidence; does not relax data/hash/completeness checks')
    args=parser.parse_args();args.out=args.out.resolve()
    require(Path(__file__).resolve().parent in args.out.parents,'Output must be beneath this report_tools folder')
    require(not args.out.exists(),'Refusing to overwrite analysis output')
    plan,articles,rr,gr,chunks,provenance=load_runs(args.retrieval_run.resolve(),args.generation_run.resolve(),args.plan.resolve(),args.development,args.allow_unverified_upload)
    tables,article_rows,differences,primary=summarize_tables(rr,gr,articles,plan)
    tests=dict(status='not_run_development',formal_hypothesis_decisions=False) if args.development else primary_tests(primary,plan['primary_tests'])
    h3=h3_analysis(gr,articles,plan)
    if args.development:h3['status']='development_descriptive_only'
    failed=sorted({r['id'] for r in gr if r['answer_metrics']['exact_match']<1})
    selected=random.Random(plan['evaluation_design']['seed']).sample(failed,min(30,len(failed)))
    review=[]
    for qid in selected:
        for r in gr:
            if r['id']==qid:review.append(dict(id=qid,strategy=r['strategy'],article=r['article'],question=r['question'],prediction=r['prediction'],references=json.dumps(r['answers'],ensure_ascii=False),exact_match=r['answer_metrics']['exact_match'],f1=r['answer_metrics']['f1'],hit_at_3=r['retrieval_metrics']['hit_at_3'],visible_evidence_hit=r['visible_evidence_hit'],category='',cause='',annotation_pending=True))
    efficiency={s:dict(n_chunks=len(chunks[s]),word_lengths=dict(mean=float(np.mean([len(c['text'].split()) for c in chunks[s]])),maximum=max(len(c['text'].split()) for c in chunks[s])),embedding_bytes=(args.retrieval_run/s/'chunk_embeddings.npy').stat().st_size,context_truncations=sum(r['packed_prompt']['context_truncated'] for r in gr if r['strategy']==s),max_input_tokens=max(r['packed_prompt']['input_tokens'] for r in gr if r['strategy']==s),max_output_ids=max(len(r['output_token_ids']) for r in gr if r['strategy']==s)) for s in METHODS}
    for method in METHODS:
        retrieved=[r for r in rr if r['strategy']==method]; generated=[r for r in gr if r['strategy']==method]
        elapsed=[r['timing']['answer_pipeline_elapsed_s'] for r in generated if 'timing' in r]
        require(all(math.isfinite(t) and t>=0 for t in elapsed),'Invalid recorded answer-pipeline times')
        efficiency[method]['answer_pipeline_timing']=(dict(n_records=len(elapsed),total_seconds=sum(elapsed),mean_seconds=float(np.mean(elapsed)),median_seconds=float(np.median(elapsed)),max_seconds=max(elapsed)) if elapsed else dict(status='not_recorded_in_this_run'))
        efficiency[method].update(corpus_span_missing_count=sum(1-r['corpus_answer_span_coverage'] for r in retrieved),
            ranking_miss_with_span_available_count=sum(r['corpus_answer_span_coverage']==1 and r['retrieval_metrics']['hit_at_3']==0 for r in retrieved),
            evidence_removed_by_packing_count=sum(r['retrieval_metrics']['hit_at_3']==1 and r['visible_evidence_hit']==0 for r in generated),
            output_new_tokens_at_cap_count=sum(len(r['output_token_ids'])-1>=32 for r in generated),
            output_cap_interpretation='At-cap flag only; human review determines whether an answer was cut off')
    provenance.update(analysis_script_sha256=sha(__file__),analysis_python=platform.python_version(),numpy_version=np.__version__,created_at_utc=datetime.now(timezone.utc).isoformat())
    result=dict(mode=provenance['mode'],formal_hypothesis_decisions=not args.development,provenance=provenance,tables=tables,paired_differences=differences,primary_tests=tests,H3=h3,efficiency=efficiency,error_selection=dict(seed=42,available_failed_questions=len(failed),selected_question_ids=selected,review_records=len(review),annotation_status='PENDING human review'),runtime=dict(retrieval=provenance['retrieval_manifest'].get('timing',{}),generation_attempts=provenance['generation_manifest'].get('attempts',[])),limitations=['Conditional on fixed corpus and models; article bootstrap does not rebuild indexes.','Marginal intervals are not simultaneous; method-label exchangeability and paired symmetry are assumptions.','Development mode has no formal hypothesis decisions.'] )
    args.out.mkdir(parents=True)
    write(args.out/'analysis.json',result)
    for phase in ['retrieval','generation']:csv_write(args.out/(phase+'_table.csv'),tables[phase])
    for phase in ['retrieval','generation']:csv_write(args.out/(phase+'_article_metrics.csv'),[r for r in article_rows if r['phase']==phase])
    csv_write(args.out/'paired_differences.csv',differences);csv_write(args.out/'pending_error_review.csv',review)
    raw=[]
    for phase,rows in [('retrieval',rr),('generation',gr)]:
        for r in rows:raw.append(dict(phase=phase,strategy=r['strategy'],id=r['id'],article=r['article'],for_generation=r['for_generation'],hit_at_1=r['retrieval_metrics']['hit_at_1'],hit_at_3=r['retrieval_metrics']['hit_at_3'],hit_at_5=r['retrieval_metrics']['hit_at_5'],corpus_coverage=r.get('corpus_answer_span_coverage',''),exact_match=r.get('answer_metrics',{}).get('exact_match',''),f1=r.get('answer_metrics',{}).get('f1',''),visible_hit=r.get('visible_evidence_hit',''),answer_pipeline_seconds=r.get('timing',{}).get('answer_pipeline_elapsed_s','')))
    csv_write(args.out/'per_question_metrics.csv',raw)
    draw_figure(args.out/'retrieval_figure.png',tables['retrieval'],['hit_at_1','hit_at_3','hit_at_5'],'Answer-span retrieval coverage',provenance['mode'],len(articles))
    draw_figure(args.out/'generation_figure.png',tables['generation'],['exact_match','f1'],'Generated-answer accuracy',provenance['mode'],len(articles))
    write(args.out/'artifact_sha256.json',{p.name:sha(p) for p in args.out.iterdir() if p.is_file()})
    print(json.dumps(dict(mode=provenance['mode'],retrieval_questions=len(rr)//3,generation_questions=len(gr)//3,outputs=str(args.out),formal_decisions=not args.development),indent=2))


if __name__=='__main__':main()
