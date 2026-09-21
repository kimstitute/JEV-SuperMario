"""Re-ask saved observations without advancing an emulator or using future outcomes."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from .candidates import candidate_state
from .policy import JevPolicy, TokenBudgetReached
from .rulebook import DEFAULT_PROFILE, PROFILES, contract_metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path)
    p.add_argument('--frames', type=int, nargs='+')
    p.add_argument('--suite',type=Path,help='paired comparison manifest; uses all prompt profiles')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-input-tokens', type=int, default=200_000)
    p.add_argument('--prompt-profile',choices=PROFILES,default=DEFAULT_PROFILE)
    args = p.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1]/'.env')
    if args.suite:
        if args.run or args.frames:
            p.error('Use either suite or run/frames, not both.')
        from .prompt_evaluation import run_suite
        run_suite(args.suite,args.output,args.repeats,args.max_input_tokens)
        return
    if not args.run or not args.frames:
        p.error('run and frames are required without suite.')
    rows = [json.loads(line) for line in (args.run/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = [(i,r) for i,r in enumerate(rows) if r['frame'] in args.frames]
    if set(args.frames) != {r['frame'] for _,r in selected}:
        p.error('Every requested frame must have a saved decision.')
    args.output.mkdir(parents=True, exist_ok=False)
    load_dotenv(Path(__file__).resolve().parents[1]/'.env')
    tokens = 0
    calls = 0
    results = []
    def audit(event, exchange):
        nonlocal tokens, calls
        if event == 'request':
            if tokens >= args.max_input_tokens:
                raise TokenBudgetReached()
            calls += 1
        elif event == 'response':
            tokens += exchange['response'].get('usage',{}).get('input_tokens',0)
        with (args.output/'api.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps({'event':event,**exchange})+'\n')
    policy = JevPolicy(on_exchange=audit,prompt_profile=args.prompt_profile)
    try:
        for i,r in selected:
            # Only past executed controls are input; current/future outcomes never enter.
            state = candidate_state(r['raw_observations'],rows[:i])
            action, request, response, latency = policy.choose(state)
            result = {'frame':r['frame'],'old_action':r['action'],'action':action,
                      'state':state,'request':request,'response':response,
                      'api_exchanges':policy.exchanges,'latency_s':latency}
            results.append(result)
            with (args.output/'decisions.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(result)+'\n')
            print(f"frame={r['frame']} {r['action']} -> {action} interval={state['action_frames']}",flush=True)
    finally:
        policy.close()
        summary = {'evaluated_frames':len(results),'api_calls':calls,'input_tokens':tokens,
                   'prompt_contract':contract_metadata(args.prompt_profile),
                   'cost_estimate_usd':tokens/1_000_000*.042,
                   'note':'Saved-state decision comparison, not a rollout or a success-rate estimate.'}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
