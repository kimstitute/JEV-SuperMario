"""Single-call, non-pausing Jev controller.

The emulator advances one frame at a time while a Choice request is in flight.
The action returned by that request becomes the held action for the next wait
interval.  This is intentionally a separate controller so the paused two-stage
experiment remains reproducible.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import time

import gymnasium as gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from PIL import Image

from .candidates import candidate_state
from .policy import ACTIONS, JevPolicy, TokenBudgetReached
from .run import (ROOT, build_state, make_report, observe_frame, self_state,
                  write_json)
from .rulebook import canonical_hash, contract_metadata
from .vision import ScreenObserver
from .system_state import SystemStateParser


def run(args):
    out = args.output or ROOT / 'runs' / f'{time.strftime("%Y%m%d-%H%M%S")}-realtime'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    (out / 'history').mkdir()
    summary = {
        'log_schema_version': 'realtime-v2',
        'mode': 'jev', 'controller': 'realtime', 'requested_model': args.model,
        'stage': 'SuperMarioBros-1-1-v0', 'decision_limit': args.decisions,
        'action_frames': args.action_frames, 'decision_interval_frames': args.action_frames,
        'frame_sample_interval': args.frame_sample_interval,
        'max_stale_frames': args.max_stale_frames,
        'prompt_contract': contract_metadata(args.prompt_profile),
        'observation_format': 'canonical_system_state_plus_debug_analysis',
        'perception': 'emulator info + NES RAM canonical state; RGB analysis retained for debug',
        'status': 'running', 'api_calls': 0, 'input_tokens': 0, 'output_tokens': 0,
        'max_world_x': 0, 'game_frames': 0, 'decisions': 0,
        'note': 'One Jev Choice per scheduled decision. The emulator advances during API latency; '
                'late responses older than max_stale_frames are logged and discarded.',
    }
    write_json(out / 'summary.json', summary)
    # Small polling target for the optional live monitor UI.
    write_json(out / 'live_state.json', {'status': 'starting', 'summary': summary,
                                         'current_action': 'NOOP', 'jev_action': None})
    write_json(out / 'manifest.json', {
        'schema': 'jev-mario-realtime-v2',
        'controller': 'realtime', 'model': args.model,
        'prompt_profile': args.prompt_profile,
        'stage': 'SuperMarioBros-1-1-v0', 'seed': 0,
        'runtime_fps': 60, 'initial_action': 'NOOP',
        'files': {
            'decisions': 'decisions.jsonl', 'frames': 'frames.jsonl',
            'perception': 'perception.jsonl', 'api': 'api.jsonl',
            'summary': 'summary.json', 'replay': 'index.html',
        },
        'privacy': 'No API key or authorization header is logged. Canonical enemy/local-grid RAM-derived state is logged because it is the model input.',
    })
    policy = JevPolicy(args.model, timeout=args.timeout, controller='realtime',
                       prompt_profile=args.prompt_profile)
    frame = 0
    total_tokens = 0
    api_sequence = 0
    api_latencies = []
    records = []
    started = time.perf_counter()
    env = None
    try:
        def exchange(event, item):
            nonlocal total_tokens, api_sequence
            if event == 'request':
                summary['api_calls'] += 1
                api_sequence += 1
                item['api_sequence'] = api_sequence
            elif event == 'response':
                usage = item['response'].get('usage', {})
                total_tokens += int(usage.get('input_tokens', 0))
                summary['input_tokens'] = total_tokens
                summary['output_tokens'] += int(usage.get('output_tokens', 0))
                api_latencies.append(item.get('latency_s', 0.0))
                summary['mean_api_latency_s'] = round(sum(api_latencies) / len(api_latencies), 4)
            with (out / 'api.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({'game_frame': frame, 'event': event, **item},
                                     ensure_ascii=False) + '\n')
            write_json(out / 'summary.json', summary)
            write_json(out / 'live_state.json', {'status': summary.get('status'), 'summary': summary,
                                                 'current_action': current_action if 'current_action' in locals() else 'NOOP',
                                                 'jev_action': item.get('response', {}).get('answers', {}).get('action', {}).get('choice') if event == 'response' else None})

        policy.on_exchange = exchange
        env = JoypadSpace(gym.make('SuperMarioBros-1-1-v0'), list(ACTIONS.values()))
        rgb, info = env.reset(seed=0)
        mario = self_state(info, env.unwrapped.ram, None, 0, [])
        # Full template matching is expensive (~145ms/frame on this host).
        # Realtime keeps a per-frame tracked sample and refreshes appearance
        # detections every few frames.
        observer = ScreenObserver(detection_interval=6)
        system_parser = SystemStateParser()
        buffer = deque(maxlen=13)
        first_sample = observe_frame(observer, rgb, mario, 0)
        first_sample['observation']['system_state'] = system_parser.parse(
            info, env.unwrapped.ram, mario, 0, current_action if 'current_action' in locals() else 'NOOP')
        buffer.append(first_sample)
        initial_x = int(info['x_pos'])
        summary['start_world_x'] = initial_x
        summary['max_world_x'] = initial_x
        current_action = 'NOOP'
        frame_log = (out / 'frames.jsonl').open('w', encoding='utf-8')
        perception_log = (out / 'perception.jsonl').open('w', encoding='utf-8')

        def log_frame(frame_number, action, reward, info, sample, terminal=False, truncated=False):
            """Write one complete frame record; this is the primary replay timeline."""
            frame_log.write(json.dumps({
                'frame': frame_number, 'action_held': action,
                'reward': float(reward), 'world_x': int(info['x_pos']),
                'terminated': bool(terminal), 'truncated': bool(truncated),
                'mario': sample['mario'],
                'observation_ref': frame_number,
            }, ensure_ascii=False) + '\n')
            perception_log.write(json.dumps(sample['observation'], ensure_ascii=False) + '\n')

        log_frame(0, current_action, 0.0, info, buffer[-1])
        Image.fromarray(rgb).save(out / 'frames' / 'game-000000.png')
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = None
            release_edge = False
            next_decision = 0
            terminated = truncated = False
            while len(records) < args.decisions and not (terminated or truncated):
                # One request at a time, on a fixed frame cadence. The emulator
                # never waits for HTTP; the active action remains held meanwhile.
                if pending is None and frame >= next_decision:
                    state, boxes = build_state(buffer, args.action_frames)
                    policy_state = candidate_state(state, records)
                    request_frame = frame
                    image_path = f'frames/decision-{next_decision:04d}-frame-{request_frame:06d}.png'
                    Image.fromarray(buffer[-1]['rgb']).save(out / image_path)
                    pending = {
                        'future': executor.submit(policy.choose, policy_state),
                        'state': state, 'policy_state': policy_state,
                        'request_frame': request_frame, 'image': image_path,
                        'decision': next_decision, 'reward_sum': 0.0,
                        'action_during_wait': current_action,
                    }
                    next_decision = frame + args.action_frames
                frame_started = time.perf_counter()
                effective_action = 'NOOP' if release_edge else current_action
                release_edge = False
                rgb, reward, terminated, truncated, last_info = env.step(list(ACTIONS).index(effective_action))
                frame += 1
                mario = self_state(last_info, env.unwrapped.ram, mario, frame,
                                   ACTIONS[effective_action])
                buffer.append(observe_frame(observer, rgb, mario, frame))
                buffer[-1]['observation']['system_state'] = system_parser.parse(
                    last_info, env.unwrapped.ram, mario, frame, effective_action, reward)
                log_frame(frame, effective_action, reward, last_info, buffer[-1], terminated, truncated)
                # PNG encoding is sampled and never part of the critical path
                # for every frame. JSONL remains the complete replay timeline.
                if args.frame_sample_interval and frame % args.frame_sample_interval == 0:
                    Image.fromarray(rgb).save(out / f'frames/game-{frame:06d}.png')
                    Image.fromarray(rgb).save(out / 'live_frame.png')
                summary['max_world_x'] = max(summary['max_world_x'], int(last_info['x_pos']))
                if pending is not None:
                    pending['reward_sum'] += float(reward)
                if pending is not None and pending['future'].done():
                    action, payload, response, latency = pending['future'].result()
                    wait_frames = frame - pending['request_frame']
                    old_action = current_action
                    stale_discarded = wait_frames > args.max_stale_frames
                    applied_action = old_action if stale_discarded else action
                    # A held across a landing must be released for one frame
                    # before a new jump press can be recognized by the NES.
                    release_edge = ('A' in ACTIONS[applied_action] and 'A' in ACTIONS[old_action]
                                    and mario.get('grounded', False))
                    record = {
                        'decision': pending['decision'], 'frame': frame, 'action': applied_action,
                        'response_action': action,
                        'request_frame': pending['request_frame'],
                        'state_hash': canonical_hash(pending['policy_state']),
                        'api_sequence': pending['decision'] + 1,
                        'latency_s': latency, 'state': pending['policy_state'],
                        'raw_observations': pending['state'],
                        'request': payload, 'response': response,
                        'answer': response['answers']['action'], 'image': pending['image'],
                        'outcome': {
                            'executed_frames': wait_frames, 'action_during_wait': pending['action_during_wait'],
                            'reward': pending['reward_sum'], 'world_x': int(last_info['x_pos']),
                            'terminated': bool(terminated), 'truncated': bool(truncated),
                            'clear': bool(last_info.get('flag_get')),
                            'mario_after': {k: mario[k] for k in
                                            ('world_x', 'feet_y', 'vx', 'vy', 'grounded')},
                        },
                        'realtime': {'request_frame': pending['request_frame'],
                                     'response_frame': frame,
                                     'stale_state_frames': wait_frames,
                                     'response_delay_frames': wait_frames,
                                     'release_edge_inserted': release_edge,
                                     'stale_response_discarded': stale_discarded},
                    }
                    records.append(record)
                    with (out / 'decisions.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(record, ensure_ascii=False) + '\n')
                    current_action = applied_action
                    write_json(out / 'live_state.json', {
                        'status': summary.get('status', 'running'), 'summary': summary,
                        'current_action': current_action, 'jev_action': action,
                        'decision': record['decision'], 'frame': frame,
                        'latency_s': latency, 'world_x': int(last_info['x_pos']),
                        'stale_state_frames': wait_frames,
                        'jump_phase': (pending['policy_state'].get('reaction_timing') or {}).get('jump_control_phase'),
                    })
                    pending = None
                    summary.update(game_frames=frame, decisions=len(records),
                                   progress_pixels=summary['max_world_x'] - initial_x)
                    write_json(out / 'summary.json', summary)
                    make_report(out, records, summary)
                    print(f'{record["decision"]:03d} frame={frame:05d} x={last_info["x_pos"]:4d} '
                          f'action={applied_action:10s} wait_action={record["outcome"]["action_during_wait"]:10s} '
                          f'wait_frames={wait_frames} latency={latency:.3f}s', flush=True)
                    if terminated or truncated:
                        summary['status'] = 'clear' if last_info.get('flag_get') else ('truncated' if truncated else 'death')
                        break
                time.sleep(max(0.0, (1 / 60) - (time.perf_counter() - frame_started)))
            if not terminated and not truncated and len(records) >= args.decisions:
                summary['status'] = 'decision_limit'
            elif terminated or truncated:
                summary['status'] = 'clear' if last_info.get('flag_get') else ('truncated' if truncated else 'death')
        frame_log.close()
        perception_log.close()
    except TokenBudgetReached:
        summary['status'] = 'token_limit'
    except Exception as exc:
        summary['status'] = 'error'
        summary['error'] = str(exc)
        print(f'ERROR: {exc}', flush=True)
    finally:
        if env:
            env.close()
        policy.close()
        summary['wall_seconds'] = round(time.perf_counter() - started, 3)
        summary['game_seconds_at_60fps'] = round(frame / 60, 3)
        summary['progress_pixels'] = summary.get('max_world_x', initial_x if 'initial_x' in locals() else 0) - (initial_x if 'initial_x' in locals() else 0)
        summary['mean_decision_latency_s'] = round(
            sum(r['latency_s'] for r in records) / len(records), 4) if records else None
        summary['action_counts'] = {a: sum(r['action'] == a for r in records)
                                    for a in sorted({r['action'] for r in records})}
        summary['frame_log_complete'] = (summary.get('game_frames', 0) + 1)
        if 'frame_log' in locals() and not frame_log.closed:
            frame_log.close()
        if 'perception_log' in locals() and not perception_log.closed:
            perception_log.close()
        write_json(out / 'summary.json', summary)
        make_report(out, records, summary)
        print(f'REPORT: {out / "index.html"}', flush=True)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if summary['status'] == 'error' else 0
