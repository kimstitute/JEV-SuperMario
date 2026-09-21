"""Run a bounded, paused-between-decisions Mario experiment."""
import argparse
from copy import deepcopy
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import time

import gymnasium as gym
import gym_super_mario_bros  # registers environments
from nes_py.wrappers import JoypadSpace
from PIL import Image, ImageDraw
from dotenv import load_dotenv

from .policy import ACTIONS, JevPolicy, compose_answers, TokenBudgetReached
from .analysis import analyzed_state
from .candidates import candidate_state
from .rulebook import DEFAULT_PROFILE, PROFILES, contract_metadata
from .vision import ScreenObserver

ROOT = Path(__file__).resolve().parents[1]

def self_state(info, ram, previous, frame, buttons):
    # Explicit whitelist: do not expose the environment's enemy_types or map RAM.
    x = int(info['x_pos'])
    left = int(info['left_x_pos'])
    bottom = int(info['y_pixel']) + 32
    height = 16 if info['status'] == 'small' else 32
    dt = frame-previous['frame'] if previous else 0
    dx = x-previous['world_x'] if previous else 0
    dy = bottom-previous['feet_y'] if previous else 0
    return {
        'world_x':x, 'screen_x':left, 'feet_y':bottom,
        'screen_box':[left,bottom-height,16,height], 'form':info['status'],
        'vx':round(dx/dt,2) if dt else None,
        'vy':round(dy/dt,2) if dt else None,
        'grounded':int(ram[0x001d]) == 0,
        'held_buttons':list(buttons), 'frame':frame,
    }

def build_state(buffer, action_frames, spacing=4):
    latest = buffer[-1]['frame']
    wanted = {latest-i*spacing for i in range(4)}
    selected = [o for o in buffer if o['frame'] in wanted]
    history, drawings = [], []
    for sample in selected:
        obs = deepcopy(sample['observation'])
        drawings = sample['drawings']
        obs['age_frames'] = latest-sample['frame']
        history.append(obs)
    return {
        'goal':'Finish Super Mario Bros 1-1, avoiding death.',
        'action_frames':action_frames,
        'coordinates':'Pixels. Screen x right, y down. Object dx/dy relative to Mario feet center in the SAME observation. Width/height are extents from object top-left.',
        'history_spacing_frames':spacing, 'history':history,
        'recent_terrain_changes':[
            {'frame':sample['frame'],**change} for sample in buffer
            for change in sample['observation']['terrain_changes']],
    }, drawings

def observe_frame(observer, rgb, mario, frame):
    observation, drawings = observer.detect(rgb,mario,frame)
    return {'frame':frame,'rgb':rgb.copy(),'mario':deepcopy(mario),
            'observation':observation,'drawings':drawings}

def describe_observation(obs):
    """Deterministic geometry-to-language rendering, with no action advice."""
    m = obs['mario']
    width,height = m['screen_box'][2:]
    vx,vy = m['vx'],m['vy']
    motion = 'on the ground' if m['grounded'] else ('rising' if vy is not None and vy < 0 else 'falling' if vy is not None and vy > 0 else 'airborne')
    sentences = [f"Mario is {m['form']}, {width}px wide and {height}px tall, {motion}. "
                 f"Horizontal velocity: {vx} px/frame (positive=right). Vertical velocity: {vy} px/frame (positive=down). "
                 f"Currently held buttons: {', '.join(m['held_buttons']) or 'none'}."]
    for obj in obs['sprites']:
        x,y,w,h = obj['dx'],obj['dy'],obj['width'],obj['height']
        if x >= width/2:
            relation = f"to Mario's right; horizontal edge-to-edge gap is {x-width/2:g}px"
        elif x+w <= -width/2:
            relation = f"to Mario's left; horizontal edge-to-edge gap is {-width/2-x-w:g}px"
        else:
            relation = 'horizontally overlapping Mario'
        overlaps = max(y,-height) < min(y+h,0)
        source = 'Currently observed' if obj.get('source','observed')=='observed' else f"Last observed {obj['age_frames']} frames ago, current position unknown"
        sentences.append(f"{source} {obj['id']}: goomba-like ENEMY {relation}; "
                         f"its vertical span {'overlaps' if overlaps else 'does not overlap'} Mario's body. "
                         f"Its top is {y:g}px and bottom {y+h:g}px relative to Mario's feet (down positive). "
                         f"Observed world horizontal velocity: {obj['vx']} px/frame. Detection is approximate.")
    if not obs['sprites']:
        sentences.append('No enemy-like sprite was detected in this frame; detection is not exhaustive.')
    top_ground = min((o['dy'] for o in obs['terrain'] if o['kind']=='ground'),default=None)
    for obj in obs['terrain']:
        if obj['kind']=='ground' and top_ground is not None and obj['dy'] > top_ground:
            continue  # subsurface texture is redundant, not a new platform
        label = {'brick':'solid brick block','question':'solid question block',
                 'ground':'ground surface','pipe_top':'pipe top','pipe_body':'pipe body',
                 'used':'used solid block','stair':'solid stair block'}[obj['kind']]
        source = 'Observed' if obj.get('source','observed')=='observed' else f"Remembered, last seen {obj['age_frames']} frames ago ({obj['visibility']})"
        sentences.append(f"{source} {label}: horizontal interval [{obj['dx']:g}, {obj['dx']+obj['width']:g}]px "
                         f"relative to Mario feet center, top {obj['dy']:g}px relative to feet, height {obj['height']}px.")
    for gap in obs['floor_gaps']:
        sentences.append(f"Visually confirmed floor opening: from {gap['dx_start']:g} to {gap['dx_end']:g}px relative to Mario feet center.")
    return {'frame':obs['frame'],'age_frames':obs['age_frames'],'observations':sentences,
            'knowledge_grid':obs.get('knowledge_grid'), 'terrain_changes':obs.get('terrain_changes',[])}

def policy_state_for(state, observation_format):
    if observation_format == 'raw':
        return {**state,'knowledge_grid_spec':KNOWLEDGE_SPEC,
                'history':[{k:v for k,v in o.items() if not k.endswith('_grid') or k=='knowledge_grid'}
                                  for o in state['history']]}
    if observation_format == 'grid':
        return {
            'goal':state['goal'], 'action_frames':state['action_frames'],
            'grid_spec':{
                'cell_size_pixels':8,'columns':32,'rows':25,
                'origin_screen_pixels':[0,40], 'x_axis':'right', 'y_axis':'down',
                'legend':{'M':'Mario','E':'goomba-like enemy candidate',
                          '#':'ground','B':'solid brick','?':'solid question block',
                          'P':'pipe','U':'used block','S':'stair block','e':'last observed enemy position, current position unknown',
                          '.':'visually confirmed empty','X':'unknown','O':'occluded'},
                'terrain_grid':'Underlying terrain, including remembered tiles. Dot means no known terrain, NOT empty.',
                'entity_grid':'M Mario, E observed enemy, e remembered last position, dot no detected entity.',
                'knowledge_grid':KNOWLEDGE_SPEC,
                'notes':'Rows run top to bottom. Adjacent letters may be one object. Camera scrolls. RGB detections quantized to 8px, not a collision map. Offscreen is unknown.'},
            'history':[{'frame':o['frame'],'age_frames':o['age_frames'],
                        **{k:o[k] for k in ('screen_grid','terrain_grid','entity_grid','knowledge_grid','camera','terrain_changes')},
                        'tracks':o['sprites'],'mario':o['mario']}
                       for o in state['history']],
            'history_spacing_frames':state['history_spacing_frames'],
            'recent_terrain_changes':state['recent_terrain_changes'],
            'observation_format':'Separate terrain, entity and knowledge grids, plus measured tracks and Mario state.',
        }
    return {**state, 'history':[describe_observation(o) for o in state['history']],
            'knowledge_grid_spec':KNOWLEDGE_SPEC,
            'observation_format':'Measured geometry rendered as English facts. No action recommendations. Last history item is current.'}

KNOWLEDGE_SPEC = {'cell_pixels':8,'origin_screen':[0,40],
                  'legend':{'.':'visually confirmed empty cell; invisible blocks cannot be ruled out',
                            'T':'currently observed terrain','R':'remembered terrain, unconfirmed now',
                            'O':'occluded by observed Mario/enemy','X':'unknown or unsupported pixels'},
                  'rule':'Unknown and occluded cells are not evidence of a pit. Terrain memory is a separate layer.'}

def annotate(rgb, mario, objects):
    im = Image.fromarray(rgb).resize((768,720), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    for item in objects:
        x,y,w,h = item['box']
        color = '#ffaa00' if item.get('source')=='remembered' else '#ff4488' if 'candidate' in item['kind'] else '#00ffff'
        draw.rectangle((x*3,y*3,(x+w)*3,(y+h)*3),outline=color,width=2)
        draw.text((x*3,max(100,y*3-13)),item.get('id',item['kind']),fill=color,stroke_width=1,stroke_fill='black')
    x,y,w,h = mario['screen_box']
    draw.rectangle((x*3,y*3,(x+w)*3,(y+h)*3),outline='#ffffff',width=2)
    return im

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def make_report(out, records, summary):
    data = json.dumps({'summary':summary,'decisions':records}, ensure_ascii=False).replace('</','<\\/')
    template = (ROOT/'jev_mario/report.html').read_text(encoding='utf-8')
    (out/'index.html').write_text(template.replace('__RUN_DATA__',data),encoding='utf-8')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['jev','smoke'],default='jev')
    p.add_argument('--decisions',type=int,default=120)
    p.add_argument('--action-frames',type=int,default=6)
    p.add_argument('--model',default='jev-1.13.0')
    p.add_argument('--observation-format',choices=['raw','described','grid'],default='described')
    p.add_argument('--controller',choices=['realtime'],default='realtime',
                   help='Realtime is the supported controller: one Choice request while the emulator keeps running')
    p.add_argument('--prompt-profile',choices=PROFILES,default=DEFAULT_PROFILE,
                   help='candidate question/rule profile; perception and action timing remain fixed')
    p.add_argument('--max-input-tokens',type=int,default=1_000_000)
    p.add_argument('--timeout',type=float,default=40,
                   help='HTTP timeout in seconds for the realtime worker')
    p.add_argument('--frame-sample-interval',type=int,default=8,
                   help='realtime PNG sample interval; JSONL still records every frame')
    p.add_argument('--max-stale-frames',type=int,default=30,
                   help='realtime responses older than this are logged but not applied')
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    if (args.decisions < 1 or not 1 <= args.action_frames <= 24 or args.max_input_tokens < 1
            or args.frame_sample_interval < 0 or args.max_stale_frames < 1):
        p.error('decisions/tokens must be positive; action-frames must be 1..24')
    load_dotenv(ROOT/'.env')
    if args.controller == 'realtime':
        from .realtime import run as run_realtime
        return run_realtime(args)
    policy = JevPolicy(args.model,controller=args.controller,prompt_profile=args.prompt_profile) if args.mode == 'jev' else None
    out = args.output or ROOT/'runs'/f'{datetime.now():%Y%m%d-%H%M%S}-{args.mode}'
    out.mkdir(parents=True,exist_ok=False)
    (out/'frames').mkdir()
    (out/'observations').mkdir()
    (out/'history').mkdir()
    env = None
    records, replay, total_tokens = [], [], 0
    frame = 0
    started = time.perf_counter()
    summary = {'mode':args.mode,'requested_model':args.model if policy else None,
               'stage':'SuperMarioBros-1-1-v0','decision_limit':args.decisions,
               'action_frames':args.action_frames,'history_frames':4,'spacing_frames':4,
               'controller':args.controller,
               'prompt_contract':contract_metadata(args.prompt_profile) if args.controller=='candidates' else None,
               'observation_format':'candidate_consequences' if args.controller=='candidates' else 'analyzed' if args.controller=='composed' else args.observation_format,
               'perception':'screen RGB + Mario-only state; no enemy/map RAM',
               'status':'running','api_calls':0,'input_tokens':0,'output_tokens':0,
               'max_world_x':0,'game_frames':0,'decisions':0,
               'cost_estimate_usd':0, 'cost_rate_usd_per_million_input':0.042,
               'note':'SMOKE uses a scripted controller and is NOT Jev performance.' if not policy else 'Jev judgments compose buttons. Rule analysis uses RGB measurements; no other AI, hidden RAM, emulator rollout or fallback controller.'}
    write_json(out/'summary.json',summary)
    api_latencies = []
    def on_exchange(event, exchange):
        nonlocal total_tokens
        if event == 'request':
            if total_tokens >= args.max_input_tokens:
                raise TokenBudgetReached('Input-token budget reached before next API stage.')
            summary['api_calls'] += 1
        elif event == 'response':
            usage = exchange['response'].get('usage', {})
            total_tokens += int(usage.get('input_tokens', 0))
            summary['input_tokens'] = total_tokens
            summary['output_tokens'] += int(usage.get('output_tokens', 0))
            summary['actual_model'] = exchange['response'].get('model')
            summary['cost_estimate_usd'] = total_tokens/1_000_000*.042
            api_latencies.append(exchange['latency_s'])
        with (out/'api.jsonl').open('a', encoding='utf-8') as api_log:
            api_log.write(json.dumps({'event':event,'game_frame':frame,**exchange}, ensure_ascii=False)+'\n')
        write_json(out/'summary.json',summary)
    if policy:
        policy.on_exchange = on_exchange
    try:
        env = JoypadSpace(gym.make('SuperMarioBros-1-1-v0'), list(ACTIONS.values()))
        rgb, info = env.reset(seed=0)
        mario = self_state(info,env.unwrapped.ram,None,0,[])
        buffer = deque(maxlen=13)
        observer = ScreenObserver()
        buffer.append(observe_frame(observer,rgb,mario,0))
        initial_x = int(info['x_pos'])
        summary['start_world_x'] = initial_x
        summary['max_world_x'] = initial_x
        replay.append(Image.fromarray(rgb.copy()).resize((512,480),Image.Resampling.NEAREST))
        with (out/'decisions.jsonl').open('a',encoding='utf-8') as log, (out/'perception.jsonl').open('w',encoding='utf-8') as perception_log:
            perception_log.write(json.dumps(buffer[-1]['observation'])+'\n')
            for decision in range(args.decisions):
                if policy and total_tokens >= args.max_input_tokens:
                    summary['status'] = 'token_limit'
                    break
                state, boxes = build_state(buffer,args.action_frames)
                policy_state = (candidate_state(state,records) if args.controller=='candidates' else
                                analyzed_state(state,records) if args.controller=='composed' else
                                policy_state_for(state,args.observation_format))
                history_images = []
                for obs in state['history']:
                    sample = next(s for s in buffer if s['frame']==obs['frame'])
                    path = f'history/{obs["frame"]:05d}.png'
                    Image.fromarray(sample['rgb']).save(out/path)
                    history_images.append({'frame':obs['frame'],'age_frames':obs['age_frames'],'image':path})
                raw_path = f'frames/{decision:04d}.png'
                obs_path = f'observations/{decision:04d}.png'
                Image.fromarray(buffer[-1]['rgb']).save(out/raw_path)
                annotate(buffer[-1]['rgb'],mario,boxes).save(out/obs_path)
                # Save attempted input before networking, including if the call fails.
                write_json(out/'pending-state.json',policy_state)
                if policy:
                    action, payload, response, latency = policy.choose(policy_state)
                    composition = compose_answers(response,policy_state) if args.controller=='composed' else None
                    answer = response['answers']['movement' if args.controller=='composed' else 'action']
                else:
                    # Diagnostic only: hold run+jump for 18 frames then release A
                    # for 6 frames. Never used by the Jev controller.
                    action = 'RIGHT_A_B' if decision % 4 != 3 else 'RIGHT_B'
                    payload = None
                    response = None
                    latency = 0
                    answer = {'choice':action,'confidence':None,'probabilities':{}}
                    composition = None
                record = {'decision':decision,'frame':frame,'action':action,
                          'latency_s':latency,'state':policy_state,'raw_observations':state,
                          'history_images':history_images,'request':payload,
                          'response':response,'answer':answer,
                          'api_exchanges':deepcopy(policy.exchanges) if policy else [],
                          'composition':composition,
                          'image':raw_path,'observation_image':obs_path}
                action_index = list(ACTIONS).index(action)
                reward_sum = 0
                terminated = truncated = False
                actual_frames = 0
                landed = False
                for _ in range(policy_state['action_frames']):
                    was_grounded = mario['grounded']
                    rgb, reward, terminated, truncated, info = env.step(action_index)
                    frame += 1
                    actual_frames += 1
                    reward_sum += float(reward)
                    mario = self_state(info,env.unwrapped.ram,mario,frame,ACTIONS[action])
                    buffer.append(observe_frame(observer,rgb,mario,frame))
                    perception_log.write(json.dumps(buffer[-1]['observation'])+'\n')
                    summary['perception_updates'] = observer.updates
                    summary['max_world_x'] = max(summary['max_world_x'], int(info['x_pos']))
                    if frame % 3 == 0 or terminated or truncated:
                        replay.append(Image.fromarray(rgb.copy()).resize((512,480),Image.Resampling.NEAREST))
                    if terminated or truncated:
                        break
                    if args.controller!='direct' and not was_grounded and mario['grounded']:
                        landed = True
                        break
                record['outcome'] = {'executed_frames':actual_frames,'reward':reward_sum,
                                     'world_x':int(info['x_pos']),'terminated':bool(terminated),
                                     'truncated':bool(truncated),'clear':bool(info.get('flag_get')),
                                     'early_decision_on_landing':landed,
                                     'mario_after':{k:mario[k] for k in ('world_x','feet_y','vx','vy','grounded')}}
                records.append(record)
                log.write(json.dumps(record,ensure_ascii=False)+'\n')
                log.flush()
                perception_log.flush()
                summary.update(game_frames=frame,decisions=len(records),
                               progress_pixels=summary['max_world_x']-initial_x)
                write_json(out/'summary.json',summary)
                make_report(out,records,summary)
                print(f'{decision:03d} frame={frame:04d} x={info["x_pos"]:4d} action={action:10s} '
                      f'confidence={answer.get("confidence")} latency={latency:.2f}s '
                      f'tokens={total_tokens}',flush=True)
                if terminated or truncated:
                    summary['status'] = 'clear' if info.get('flag_get') else ('truncated' if truncated else 'death')
                    break
            else:
                summary['status'] = 'decision_limit'
    except TokenBudgetReached:
        summary['status'] = 'token_limit'
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
    except Exception as exc:
        summary['status'] = 'error'
        # Our API exceptions omit response bodies and headers.
        summary['error'] = str(exc)
        print(f'ERROR: {exc}',flush=True)
    finally:
        if env:
            env.close()
        if policy:
            policy.close()
        summary['wall_seconds'] = round(time.perf_counter()-started,3)
        summary['game_seconds_at_60fps'] = round(frame/60,3)
        latencies = [r['latency_s'] for r in records]
        summary['mean_decision_latency_s'] = round(sum(latencies)/len(latencies),3) if latencies else None
        summary['mean_api_latency_s'] = round(sum(api_latencies)/len(api_latencies),3) if api_latencies else None
        if replay:
            replay[0].save(out/'replay.webp',save_all=True,append_images=replay[1:],
                           duration=50,loop=0,lossless=True)
        write_json(out/'summary.json',summary)
        make_report(out,records,summary)
        print(f'REPORT: {out / "index.html"}',flush=True)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 1 if summary['status'] == 'error' else 0

if __name__ == '__main__':
    raise SystemExit(main())
