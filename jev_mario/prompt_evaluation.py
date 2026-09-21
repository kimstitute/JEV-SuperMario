"""Paired, repeated prompt ablations over frozen real observations."""
from copy import deepcopy
import json
from pathlib import Path
from .candidates import candidate_state
from .policy import JevPolicy, TokenBudgetReached
from .rulebook import PROFILES, canonical_hash, contract_metadata


def load_cases(manifest):
    manifest=Path(manifest)
    root=Path(__file__).resolve().parents[1]
    cases=json.loads(manifest.read_text(encoding='utf-8'))['cases']
    loaded={}
    result=[]
    if len({c['id'] for c in cases})!=len(cases):
        raise ValueError('Case IDs must be unique.')
    for case in cases:
        if 'frozen_state' in case:
            state=json.loads((root/case['frozen_state']).read_text(encoding='utf-8'))
            digest=canonical_hash(state)
            if digest!=case['base_state_sha256']:
                raise ValueError(f"Frozen state hash mismatch: {case['id']}")
            result.append({**case,'base_state':state})
            continue
        path=root/case['run']/'decisions.jsonl'
        if path not in loaded:
            loaded[path]=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        rows=loaded[path]
        i=next(i for i,r in enumerate(rows) if r['frame']==case['frame'])
        row=rows[i]
        # A probe already includes prefix history in its saved candidate state.
        # Rebuilding it from the probe's local rows would silently drop that history.
        state=(deepcopy(row['state']) if 'candidates' in row['state'] else
               candidate_state(row['raw_observations'],rows[:i]))
        result.append({**case,'base_state':state,'base_state_sha256':canonical_hash(state),
                       'historical_action':row['action']})
    return result


def run_suite(manifest, output, repeats=2, max_input_tokens=1_500_000):
    if not 1<=repeats<=5:
        raise ValueError('Use 1..5 repeats.')
    cases=load_cases(manifest)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'cases').mkdir()
    for c in cases:
        (output/'cases'/f"{c['id']}.json").write_text(json.dumps(c,indent=2),encoding='utf-8')
    records=[]
    summary={'status':'running','model':'jev-1.13.0','case_count':len(cases),
             'repeats':repeats,'api_calls':0,'input_tokens':0,'output_tokens':0,
             'profiles':{p:{'contract':contract_metadata(p),'decisions':0,'api_calls':0,
                            'input_tokens':0,'output_tokens':0,'api_seconds':0.}
                         for p in PROFILES},
             'note':'Actions and judgments compared on identical inputs, not labeled accuracy or game success.'}
    context={}
    def audit(event,exchange):
        metrics=summary['profiles'][context['profile']]
        if event=='request':
            if summary['input_tokens']>=max_input_tokens:
                raise TokenBudgetReached()
            summary['api_calls']+=1
            metrics['api_calls']+=1
        elif event=='response':
            for name in ('input_tokens','output_tokens'):
                n=exchange['response'].get('usage',{}).get(name,0)
                summary[name]+=n
                metrics[name]+=n
            metrics['api_seconds']+=exchange['latency_s']
        with (output/'api.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({**context,'event':event,**exchange})+'\n')
    try:
        for case_index,c in enumerate(cases):
            for repeat in range(repeats):
                shift=(case_index+repeat)%len(PROFILES)
                order=PROFILES[shift:]+PROFILES[:shift]
                for profile in order:
                    context.update(case=c['id'],repeat=repeat,profile=profile)
                    policy=JevPolicy(prompt_profile=profile,on_exchange=audit)
                    try:
                        action,request,response,latency=policy.choose(c['base_state'])
                    finally:
                        policy.close()
                    if canonical_hash(c['base_state'])!=c['base_state_sha256']:
                        raise AssertionError('An evaluation mutated the frozen base state.')
                    r={**context,'base_state_sha256':c['base_state_sha256'],
                       'action':action,'request':request,'response':response,
                       'api_exchanges':policy.exchanges,'latency_s':latency}
                    records.append(r)
                    summary['profiles'][profile]['decisions']+=1
                    with (output/'decisions.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(r)+'\n')
                    print(f"{c['id']} repeat={repeat} {profile}: {action}",flush=True)
        summary['status']='complete'
    except TokenBudgetReached:
        summary['status']='token_limit'
    except Exception as exc:
        summary.update(status='error',error=str(exc))
        raise
    finally:
        summary['cost_estimate_usd']=summary['input_tokens']/1_000_000*.042
        for values in summary['profiles'].values():
            values['cost_estimate_usd']=values['input_tokens']/1_000_000*.042
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        table=['# Rule prompt comparison','',
               'Identical saved states; actions alone are not success labels. Each cell lists repeats.',
               '', '| Case | legacy-v2 | rules-only-v1 | rules-v1 |', '|---|---|---|---|']
        for c in cases:
            cells=[', '.join(r['action'] for r in records if r['case']==c['id'] and r['profile']==p)
                   or 'incomplete' for p in PROFILES]
            table.append('| '+c['id']+' | '+' | '.join(cells)+' |')
        table += ['',f"Status: {summary['status']}; API calls: {summary['api_calls']}; "
                  f"input tokens: {summary['input_tokens']}; estimated cost: ${summary['cost_estimate_usd']:.6f}"]
        (output/'comparison.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
        print(json.dumps(summary),flush=True)
    return summary
