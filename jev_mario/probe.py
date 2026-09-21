"""Reproduce a recorded decision, then test the new controller in the real game.

Prefix actions are setup only, never counted as new-controller performance.
All new inputs still come from RGB + Mario-only state; no future rollout is input.
"""
import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import time
import gymnasium as gym
import gym_super_mario_bros  # environment registration
from nes_py.wrappers import JoypadSpace
from dotenv import load_dotenv
from PIL import Image
from .run import ROOT, self_state, observe_frame, build_state, annotate, write_json, make_report
from .vision import ScreenObserver
from .policy import ACTIONS, JevPolicy, TokenBudgetReached
from .candidates import candidate_state
from .rulebook import DEFAULT_PROFILE, PROFILES, contract_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--decisions', type=int, default=30)
    parser.add_argument('--prompt-profile',choices=PROFILES,default=DEFAULT_PROFILE)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    prior = [json.loads(x) for x in (args.run/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    target = next((r for r in prior if r['frame']==args.frame), None)
    if target is None or not 1 <= args.decisions <= 60:
        parser.error('Use an existing decision frame and 1..60 decisions.')
    out = args.output or ROOT/'runs'/f'{datetime.now():%Y%m%d-%H%M%S}-matched-probe-{args.prompt_profile}'
    out.mkdir(parents=True,exist_ok=False)
    for name in ('frames','observations','history'):
        (out/name).mkdir()
    load_dotenv(ROOT/'.env')
    summary = {'mode':'jev','controller':'candidates','probe':True,'status':'setup',
               'prompt_contract':contract_metadata(args.prompt_profile),
               'setup_source':str(args.run),'setup_target_frame':args.frame,
               'api_calls':0,'input_tokens':0,'output_tokens':0,'cost_estimate_usd':0,
               'note':'Recorded actions reproduce the initial scene only. Progress is measured AFTER setup; this is not a level-start run.'}
    records, replay, api_times = [], [], []
    frame = 0
    started = time.perf_counter()
    def audit(event, exchange):
        if event == 'request':
            if summary['input_tokens'] >= 500_000:
                raise TokenBudgetReached()
            summary['api_calls'] += 1
        elif event == 'response':
            for name in ('input_tokens','output_tokens'):
                summary[name] += exchange['response'].get('usage',{}).get(name,0)
            summary['cost_estimate_usd'] = summary['input_tokens']/1_000_000*.042
            api_times.append(exchange['latency_s'])
        with (out/'api.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps({'event':event,'game_frame':frame,**exchange})+'\n')
    policy = JevPolicy(on_exchange=audit,prompt_profile=args.prompt_profile)
    env = JoypadSpace(gym.make('SuperMarioBros-1-1-v0'),list(ACTIONS.values()))
    try:
        rgb,info = env.reset(seed=0)
        mario = self_state(info,env.unwrapped.ram,None,0,[])
        observer = ScreenObserver()
        buffer = deque([observe_frame(observer,rgb,mario,0)],maxlen=13)
        # Use only executed actions strictly before the target decision.
        for r in prior:
            if r['frame'] >= args.frame:
                break
            if r['frame'] != frame:
                raise ValueError('Recorded prefix is not contiguous.')
            for _ in range(r['outcome']['executed_frames']):
                rgb,_,terminated,truncated,info = env.step(list(ACTIONS).index(r['action']))
                frame += 1
                mario = self_state(info,env.unwrapped.ram,mario,frame,ACTIONS[r['action']])
                buffer.append(observe_frame(observer,rgb,mario,frame))
                if terminated or truncated:
                    raise ValueError('Prefix terminated before target.')
        expected = target['raw_observations']['history'][-1]['mario']
        if frame != args.frame or any(mario[k] != expected[k] for k in
                ('world_x','feet_y','screen_box','grounded','held_buttons','vx','vy')):
            raise ValueError('Reproduced Mario state differs from saved target; no Jev action executed.')
        # RGB must also match the original image, independent of self-state equality.
        import numpy as np
        original_rgb = np.asarray(Image.open(args.run/target['image']).convert('RGB'))
        if not np.array_equal(rgb,original_rgb):
            raise ValueError('Reproduced RGB differs from saved target; no Jev action executed.')
        summary.update(status='running',setup_rgb_exact_match=True,start_world_x=mario['world_x'],
                       max_world_x=mario['world_x'],progress_pixels=0)
        past = [r for r in prior if r['frame'] < args.frame]
        start = mario['world_x']
        with (out/'decisions.jsonl').open('a',encoding='utf-8') as log, (out/'perception.jsonl').open('w',encoding='utf-8') as perceptions:
            perceptions.write(json.dumps(buffer[-1]['observation'])+'\n')
            for decision in range(args.decisions):
                raw,boxes = build_state(buffer,6)
                state = candidate_state(raw,past+records)
                path = f'frames/{decision:04d}.png'
                overlay = f'observations/{decision:04d}.png'
                Image.fromarray(rgb).save(out/path)
                annotate(rgb,mario,boxes).save(out/overlay)
                write_json(out/'pending-state.json',state)
                action,request,response,latency = policy.choose(state)
                record = {'decision':decision,'frame':frame,'action':action,'latency_s':latency,
                    'state':state,'raw_observations':raw,'request':request,'response':response,
                    'answer':response['answers']['action'],'composition':None,
                    'api_exchanges':deepcopy(policy.exchanges),'history_images':[],
                    'image':path,'observation_image':overlay}
                actual,landed = 0,False
                for _ in range(state['action_frames']):
                    was_grounded = mario['grounded']
                    rgb,_,terminated,truncated,info = env.step(list(ACTIONS).index(action))
                    frame += 1
                    actual += 1
                    mario = self_state(info,env.unwrapped.ram,mario,frame,ACTIONS[action])
                    buffer.append(observe_frame(observer,rgb,mario,frame))
                    perceptions.write(json.dumps(buffer[-1]['observation'])+'\n')
                    replay.append(Image.fromarray(rgb).resize((512,480),Image.Resampling.NEAREST))
                    summary['max_world_x'] = max(summary['max_world_x'],mario['world_x'])
                    landed = not was_grounded and mario['grounded']
                    if terminated or truncated or landed:
                        break
                record['outcome'] = {'executed_frames':actual,'world_x':mario['world_x'],
                    'terminated':bool(terminated),'truncated':bool(truncated),
                    'early_decision_on_landing':landed,
                    'mario_after':{k:mario[k] for k in ('world_x','feet_y','vx','vy','grounded')}}
                records.append(record)
                log.write(json.dumps(record)+'\n')
                log.flush()
                summary.update(decisions=len(records),game_frames=frame-args.frame,
                               progress_pixels=summary['max_world_x']-start)
                print(f'{decision:02d} frame={frame} x={mario["world_x"]} action={action}',flush=True)
                if terminated or truncated:
                    summary['status'] = 'clear' if info.get('flag_get') else 'death' if terminated else 'truncated'
                    break
            else:
                summary['status'] = 'decision_limit'
    except TokenBudgetReached:
        summary['status'] = 'token_limit'
    except Exception as exc:
        summary.update(status='error',error=str(exc))
        raise
    finally:
        policy.close()
        env.close()
        summary['wall_seconds'] = round(time.perf_counter()-started,3)
        summary['mean_api_latency_s'] = round(sum(api_times)/len(api_times),3) if api_times else None
        if replay:
            replay[0].save(out/'replay.webp',save_all=True,append_images=replay[1:],duration=17,loop=0,lossless=True)
        write_json(out/'summary.json',summary)
        make_report(out,records,summary)
        print(str(out),json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
