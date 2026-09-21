"""Visible-geometry consequences, adaptive horizons and coupled action candidates.

Free-space horizontal estimates are not collision simulation. Priors are explicit,
uncalibrated bounds, supplemented only by this episode's observed motion.
"""
from copy import deepcopy
import math
from .analysis import analyzed_state
from .policy import ACTIONS, DESCRIPTIONS
from . import legacy_candidate_questions as legacy_questions
from .rulebook import DEFAULT_PROFILE, REFERENCE, attach_contract

FORECAST_RULES = {
    'version': 'candidate-v2',
    'position_margin_px': 2,
    'acceleration_px_per_frame2': {
        'right': [0.04, 0.16], 'left': [0.08, 0.30], 'coast': [0.025, 0.125]},
    'run_speed_px_per_frame': 3.0, 'walk_speed_px_per_frame': 1.5,
    'assumptions': 'Initial uncalibrated horizontal bounds, not confidence intervals. '
        'No engine rollout, hidden geometry or invented vertical trajectory. '
        'Forecasts ignore collisions and are conditional on the measured motion. '
        'Sprite edges differ from collision feet; support-loss time is a range. '
        'No detected surface or ceiling means unknown, not clear space.',
}


def _motion_bounds(action, records):
    buttons = ACTIONS[action]
    kind = 'left' if 'left' in buttons else 'right' if 'right' in buttons else 'coast'
    lo, hi = FORECAST_RULES['acceleration_px_per_frame2'][kind]
    measured = []
    for r in records[-24:]:
        if r['action'] != action:
            continue
        before = r['raw_observations']['history'][-1]['mario']
        after = r['outcome'].get('mario_after', {})
        n = r['outcome']['executed_frames']
        if not (n and before['grounded'] and after.get('grounded') and
                before['vx'] is not None and after.get('vx') is not None):
            continue
        a = abs((after['vx'] - before['vx']) / n)
        # Constant speed supplies no acceleration evidence; wall impacts are excluded.
        if .02 <= a <= .35 and abs(after['world_x'] - before['world_x']) > 1:
            measured.append(a)
    # Sparse measurements widen the prior, never claim to calibrate it precisely.
    if measured:
        lo, hi = min(lo, min(measured)), max(hi, max(measured))
    return lo, hi, len(measured)


def _travel(vx, buttons, n, accel):
    target = (-3. if 'left' in buttons else
              (3. if 'B' in buttons else 1.5) if 'right' in buttons else 0.)
    x = 0.
    for _ in range(n):
        vx += max(-accel, min(accel, target - vx))
        x += vx
    return x


def _support_risk(analysis, mario):
    gap = analysis['terrain']['nearest_confirmed_gap']
    vx = mario['vx']
    if (not gap or not mario['grounded'] or vx is None or vx <= 0
            or abs(gap.get('surface_dy', 0)) > 2):
        return None
    half = mario['screen_box'][2] / 2
    edge = gap['dx_start']
    margin = FORECAST_RULES['position_margin_px']
    return {
        'cause': 'visually_confirmed_floor_opening',
        'near_edge_from_feet_center_px': edge,
        'front_edge_distance_px': gap['edge_gap_px'],
        'frames_until_feet_center_over_gap_at_current_speed': round(max(0, edge / vx), 2),
        'possible_support_loss_frames_at_current_speed': [
            round(max(0, (edge-half-margin)/vx), 2),
            round(max(0, (edge+half+margin)/vx), 2)],
        'gap_width_px': gap['visible_width_px'],
        'minimum_forward_travel_to_overlap_far_bank_px': max(0, gap['dx_end']-half),
        'forward_travel_to_center_over_far_bank_px': max(0, gap['dx_end']),
        'far_bank_height_relative_to_feet_px': gap.get('surface_dy', 0),
        'next_decision_can_be_too_late': True,
        'assumption': 'Continued rightward speed without a jump; collision-foot location is uncertain.',
    }


def _recent_jumps(records):
    """Completed observed jumps only; no replaying the emulator or fitting a full arc."""
    completed, active = [], None
    for r in records:
        before = r['raw_observations']['history'][-1]['mario']
        after = r['outcome'].get('mario_after')
        if not after:
            continue
        if before['grounded'] and not after['grounded'] and 'A' in ACTIONS[r['action']]:
            active = {'start_feet_y': before['feet_y'], 'start_x':before['world_x'],
                      'start_frame':r['frame'],
                      'start_vx':before['vx'], 'minimum_feet_y':after['feet_y'],
                      'initial_A_hold_frames':0, 'released':False}
        if active is None:
            continue
        # Use every cached sample during this jump, not just coarse action endpoints.
        for sample in r['raw_observations']['history']:
            if sample['frame'] >= active['start_frame']:
                active['minimum_feet_y'] = min(active['minimum_feet_y'], sample['mario']['feet_y'])
        active['minimum_feet_y'] = min(active['minimum_feet_y'],after['feet_y'])
        if 'A' not in ACTIONS[r['action']]:
            active['released'] = True
        elif not active['released']:
            active['initial_A_hold_frames'] += r['outcome']['executed_frames']
        if after['grounded']:
            completed.append({'start_vx':active['start_vx'],
                'initial_A_hold_frames':active['initial_A_hold_frames'],
                'observed_rise_px':active['start_feet_y']-active['minimum_feet_y'],
                'observed_forward_travel_px':after['world_x']-active['start_x'],
                'note':'Sampled completed jump; ceiling/contact and differing speed can affect the result.'})
            active = None
    return completed[-4:]


def candidate_state(state, records=()):
    base = analyzed_state(state, records)
    obs, analysis = base['observation'], base['analysis']
    m = obs['mario']
    risk = _support_risk(analysis, m)
    max_frames = state['action_frames']
    n, reasons = max_frames, []
    if risk:
        first, last = risk['possible_support_loss_frames_at_current_speed']
        if first <= max_frames * 2:
            n = min(n, 1 if first <= max_frames else 2)
            reasons.append('approaching confirmed support loss')
        risk['next_decision_can_be_too_late'] = first <= max_frames
    enemy = analysis['nearest_forward_enemy']
    contact = enemy['side_contact_in_frames_if_height_unchanged'] if enemy else None
    if contact is not None and contact <= max_frames * 2:
        n = min(n, 1 if contact <= max_frames else 2)
        reasons.append('enemy contact is close')
    obstacle = analysis['terrain']['nearest_obstacle']
    if obstacle and obstacle['edge_gap_px'] <= 24:
        n = min(n, 2)
        reasons.append('near obstacle')
    if m['grounded'] and 'A' in m['held_buttons']:
        n = 1
        reasons.append('release A after landing before a new jump')
    if not m['grounded'] and m['vy'] is not None and m['vy'] > 0:
        n = min(n, 2)
        reasons.append('descending toward landing')
    # Recompute time-dependent signals for the interval actually sent to Jev.
    if n != max_frames:
        base = analyzed_state({**state, 'action_frames': n}, records)
        analysis = base['analysis']
    candidates = {}
    system_gap_now = bool((state['history'][-1].get('system_state') or {}).get('terrain', {}).get('gap_ahead'))
    system_enemy_now = (state['history'][-1].get('system_state') or {}).get('hazard', {}).get('nearest_enemy') or {}
    urgent_jump = bool((contact is not None and contact <= max_frames * 2) or system_gap_now or
                       (m['grounded'] and 0 <= system_enemy_now.get('relative_x_pixels', 999) <= 140
                        and system_enemy_now.get('relative_velocity_x', -1) < 0))
    for action, buttons in ACTIONS.items():
        a = 'A' in buttons
        # Remove only mechanically ineffective A duplicates, never unsafe options.
        if a and ((m['grounded'] and 'A' in m['held_buttons'] and not urgent_jump) or
                  (not m['grounded'] and 'A' not in m['held_buttons'])):
            continue
        jump = 'start' if a and m['grounded'] else 'hold' if a else 'release'
        lo, hi, samples = _motion_bounds(action, records)
        vx = m['vx']
        travel = None if vx is None else sorted([
            round(_travel(vx, buttons, n, lo), 2),
            round(_travel(vx, buttons, n, hi), 2)])
        braking = None
        if 'left' in buttons and vx is not None and vx > 0:
            braking = [round(vx*vx/(2*hi), 2), round(vx*vx/(2*lo), 2)]
        support = 'unknown'
        if risk and travel is not None:
            edge = risk['near_edge_from_feet_center_px']
            support = ('jump_initiated_support_transfers_to_air' if jump == 'start' else
                       'feet_center_may_cross_gap_edge' if travel[1] >= edge-2 else
                       'feet_center_stays_before_gap_edge_this_interval')
        clearance = analysis['terrain']['headroom_now_px']
        rising = not m['grounded'] and m['vy'] is not None and m['vy'] < 0
        remaining_rise = max(0, obstacle['height_above_feet_px']) if obstacle else None
        vertical_effect = (
            'Starts a jump. A may need to stay held through later decisions to attain enough height.' if jump == 'start' else
            'CONTINUES the existing ascent; this is NOT a second jump. Keeping A held can gain more height and airtime.' if jump == 'hold' and rising else
            'A is held, but ascent is not confirmed; holding does not create a second jump.' if jump == 'hold' else
            'RELEASES A during ascent, shortening this jump. After release, pressing A again cannot restore the lost height before landing.' if rising and 'A' in m['held_buttons'] else
            'A stays released; gravity and existing motion continue.')
        candidates[action] = {
            'buttons': buttons, 'meaning': DESCRIPTIONS[action]+' '+vertical_effect, 'jump_effect': jump,
            'execute_frames': n,
            'free_space_horizontal_travel_px_range': travel,
            'acceleration_bounds': [lo, hi], 'recent_acceleration_samples': samples,
            'approximate_braking_distance_px_range': braking,
            'support_consequence': support,
            'headroom_now_px': clearance,
            'remaining_rise_to_clear_nearest_obstacle_px': remaining_rise,
            'jump_reach': 'Not predicted. Use required distance, landing height and observed outcomes; '
                          'starting a jump does not guarantee crossing.',
            'ceiling_contact_possible': (clearance is not None) if a else None,
        }
    base['analysis']['support_loss'] = risk
    base['analysis']['recent_completed_jumps'] = _recent_jumps(records)
    # Make network delay a first-class part of the observation. The controller
    # never asks Jev to infer collision timing from a stale pixel distance.
    latency_samples = [float(r.get('latency_s', 0.0)) for r in records[-8:]
                       if r.get('latency_s') is not None]
    latency_s = (sum(latency_samples) / len(latency_samples)) if latency_samples else 0.0
    response_delay_frames = round(latency_s * 60, 1)
    # `analysis` may have been recomputed for a shorter local horizon above;
    # derive timing from that final analysis, not the pre-recompute value.
    final_enemy = analysis.get('nearest_forward_enemy')
    final_contact = (final_enemy.get('side_contact_in_frames_if_height_unchanged')
                     if final_enemy else None)
    enemy_contact_after_delay = (round(final_contact - response_delay_frames, 1)
                                 if final_contact is not None else None)
    gap_edge_after_delay = (round(risk['frames_until_feet_center_over_gap_at_current_speed'] - response_delay_frames, 1)
                            if risk and risk.get('frames_until_feet_center_over_gap_at_current_speed') is not None else None)
    base['reaction_timing'] = {
        'estimated_api_latency_s': round(latency_s, 4),
        'estimated_response_delay_frames': response_delay_frames,
        'decision_horizon_frames': n,
        'jump_start_lead_frames': 12,
        'contact_within_reaction_horizon': bool(final_contact is not None and
                                                final_contact <= response_delay_frames + n),
        'jump_must_start_this_decision': bool(m['grounded'] and final_contact is not None and
                                              final_contact <= response_delay_frames + n + 12 and
                                              'A' not in m['held_buttons']),
        'jump_or_hold_must_this_decision': bool(final_contact is not None and
                                                final_contact <= response_delay_frames + n + 12),
        'takeoff_window_already_missed': bool(m['grounded'] and enemy_contact_after_delay is not None and
                                             enemy_contact_after_delay <= 0),
        'will_land_before_contact': None,
        'projected_enemy_contact_frames': enemy_contact_after_delay,
        'projected_gap_edge_frames': gap_edge_after_delay,
        'source': 'measured_recent_api_latency_and_current_motion; approximate',
    }
    system_state = obs.get('system_state') or {}
    system_terrain = system_state.get('terrain') or {}
    system_hazard = system_state.get('hazard') or {}
    system_enemy = system_hazard.get('nearest_enemy') or {}
    # Explicit control phase prevents the realtime policy from treating every
    # airborne frame as a fresh jump opportunity.  A can only start on the
    # grounded phase; while descending it is a landing decision, not a jump
    # restart decision.
    if m['grounded']:
        jump_control_phase = 'grounded_ready'
    elif m.get('vy') is not None and m.get('vy') < 0:
        jump_control_phase = 'airborne_rising'
    else:
        jump_control_phase = 'airborne_descending'
    base['reaction_timing']['jump_control_phase'] = jump_control_phase
    base['reaction_timing']['new_jump_allowed'] = bool(m['grounded'] and 'A' not in m['held_buttons'])
    system_gap = bool(system_terrain.get('gap_ahead'))
    canonical_enemy_prepare = bool(
        m['grounded'] and system_enemy.get('relative_x_pixels') is not None
        and 0 <= system_enemy.get('relative_x_pixels', 999) <= 140
        and system_enemy.get('relative_velocity_x', -1) < 0)
    if system_gap:
        # The RAM collision grid is canonical for navigation. A gap within the
        # local grid is a jump commitment even when RGB geometry is occluded.
        base['reaction_timing']['canonical_gap_ahead'] = True
        base['reaction_timing']['canonical_gap_distance_tiles'] = system_terrain.get('gap_distance_tiles')
        base['reaction_timing']['jump_or_hold_must_this_decision'] = True
        base['reaction_timing']['jump_must_start_this_decision'] = bool(
            m['grounded'] and 'A' not in m['held_buttons'])
    else:
        base['reaction_timing']['canonical_gap_ahead'] = False
    if canonical_enemy_prepare:
        base['reaction_timing']['canonical_enemy_prepare'] = True
        base['reaction_timing']['jump_or_hold_must_this_decision'] = True
        base['reaction_timing']['jump_must_start_this_decision'] = 'A' not in m['held_buttons']
    else:
        base['reaction_timing']['canonical_enemy_prepare'] = False
    # Stable semantic aliases mirror the compact real-time schema used by the
    # public Mario implementation; verbose RGB/debug fields remain in logs.
    base['player'] = deepcopy(m)
    base['trajectory'] = {
        'phase': analysis['motion']['phase'],
        'vx_px_per_frame': m.get('vx'), 'vy_px_per_frame': m.get('vy'),
        'grounded': bool(m.get('grounded')),
        'crossing_known_gap': bool(risk),
    }
    base['hazard'] = {
        'upcoming_enemies': deepcopy(analysis.get('enemies', [])[:3]),
        'nearest_enemy': deepcopy(final_enemy),
        'contact_within_reaction_horizon': base['reaction_timing']['contact_within_reaction_horizon'],
        'jump_must_start_this_decision': base['reaction_timing']['jump_must_start_this_decision'],
        'jump_or_hold_must_this_decision': base['reaction_timing']['jump_or_hold_must_this_decision'],
        'canonical_gap_ahead': base['reaction_timing']['canonical_gap_ahead'],
    }
    base['terrain'] = deepcopy(analysis['terrain'])
    base['episode'] = {
        'world_x': m.get('world_x'), 'progress_px': m.get('world_x'),
        'stalled_frames': sum(1 for r in records[-4:] if r.get('outcome', {}).get('world_x') == m.get('world_x')),
    }
    # Keep item/block evidence explicit after the wire observation is compacted.
    current_obs = state['history'][-1]
    terrain = current_obs.get('terrain', [])
    base['perception'] = {
        'question_blocks': [deepcopy(o) for o in terrain if o.get('kind') == 'question'],
        'used_blocks': [deepcopy(o) for o in terrain if o.get('kind') == 'used'],
        'bricks': [deepcopy(o) for o in terrain if o.get('kind') == 'brick'],
        'pipes': [deepcopy(o) for o in terrain if o.get('kind') in ('pipe_top', 'pipe_body')],
        'stairs': [deepcopy(o) for o in terrain if o.get('kind') == 'stair'],
        'observed_enemies': deepcopy(current_obs.get('sprites', [])),
        'observation_reliability': {
            'current_frame': 'measured RGB detections; partial occlusion possible',
            'airborne_terrain': 'prefer last_grounded_preview for support and gaps',
        },
    }
    grounded_prior = next((o for o in reversed(state['history'][:-1])
                           if o.get('mario', {}).get('grounded')), None)
    if grounded_prior:
        base['perception']['last_grounded_preview'] = {
            'frame': grounded_prior['frame'],
            'age_frames': state['history'][-1]['frame'] - grounded_prior['frame'],
            'terrain': [deepcopy(o) for o in grounded_prior.get('terrain', [])],
            'floor_gaps': [deepcopy(o) for o in grounded_prior.get('floor_gaps', [])],
        }
    else:
        base['perception']['last_grounded_preview'] = None
    base['forecast_rules'] = deepcopy(FORECAST_RULES)
    base['decision_schedule'] = {'maximum_frames': max_frames, 'execute_frames': n,
                                 'reasons': reasons or ['no nearby measured hazard'],
                                 'early_recheck_on_landing': True,
                                 'fixed_realtime_interval': max_frames}
    base['candidates'] = candidates
    # Keep focused observed facts + analyzed surfaces rather than repeated tile rows.
    base['observation'] = {k: obs[k] for k in ('frame', 'mario', 'camera', 'system_state') if k in obs}
    base['perception_source'] = 'emulator_info_and_nes_ram_canonical; RGB analysis retained for debug comparison'
    base['observation_format'] = ('Canonical emulator_info_and_nes_ram state with local collision grid; '
        'RGB-derived analysis and raw detections are retained only for debug comparison.')
    return base




def assessment_request(state, model, profile=DEFAULT_PROFILE):
    request = legacy_questions.assessment_request(attach_contract(state,profile), model)
    if profile == 'legacy-v2':
        return request
    for name, question in zip(state['candidates'],request['questions'].values()):
        if profile == 'rules-only-v1':
            question['instructions'] = REFERENCE + question['instructions']
        else:
            question['instructions'] = REFERENCE + (
                f'For candidate `{name}` in `candidates.{name}`, does executing its exact '
                'buttons for execute_frames preserve a viable local continuation under '
                '`decision_policy.assessment`? Evaluate this candidate only, using the '
                '48-frame bound and the supplied geometry, motion and past outcomes. '
                'A harmless solid-wall stop is not death. Enemy side contact differs '
                'from a descending stomp. Consider loss of a jump opportunity, not just '
                'whether Mario is alive at the end of this very short interval. '
                'Do not grade speed or preference in this question. Other questions '
                'cannot supply missing answers to this one.')
            question['criteria'] = {
                'true': 'The candidate preserves a physically plausible continuation avoiding '
                        'lethal contact or an unrecoverable fall within the defined local horizon.',
                'false': 'The candidate does not preserve such a continuation: lethal contact, '
                         'an unrecoverable fall or loss of the necessary recovery opportunity '
                         'is expected within the defined local horizon.'}
    return request


def selection_request(state, assessments, model, profile=DEFAULT_PROFILE):
    request = legacy_questions.selection_request(attach_contract(state,profile),assessments,model)
    question = request['questions']['action']
    if profile == 'rules-only-v1':
        question['instructions'] = REFERENCE + question['instructions']
    elif profile == 'rules-v1':
        question['instructions'] = REFERENCE + (
            'Choose ONE complete candidate button combination now. Use '
            '`decision_policy.selection_priority` in order and '
            '`jev_candidate_assessments` from the preceding request as fallible evidence. '
            'Both requests describe the same paused game state. Compare exact button '
            'effects, current speed, solid and enemy contact, jump hold, required height '
            'and landing support. If a reachable question block or revealed item can be '
            'collected safely, prefer that before ordinary forward progress. If a Goomba-like '
            'enemy can be stomped safely from above with a recoverable landing, prefer that '
            'after the item opportunity. Do not select waiting merely because it moves less; '
            'it needs a concrete safety or positioning benefit. Preserve enough '
            'height and motion to reach a surface from above. Avoid a confirmed gap '
            'before the opportunity to jump is lost. When viable actions differ in '
            'progress, favor the one advancing toward the right-hand goal. If every '
            'option is poor, select the most recoverable listed action.')
    return request


def realtime_request(state, model, profile=DEFAULT_PROFILE):
    """One-call candidate Choice for the non-pausing controller.

    Candidate consequences are deterministic application state.  Jev selects one
    complete action directly; it is deliberately not presented with fabricated
    per-candidate Noul answers from a preceding request.
    """
    realtime_state = deepcopy(state)
    realtime_state['execution_mode'] = 'realtime_choice_one_call'
    realtime_state['canonical_system_state'] = deepcopy(
        realtime_state.get('observation', {}).get('system_state'))
    # Keep the live action set small and stable. LEFT_A remains available to
    # the paused research controller but is intentionally excluded here.
    realtime_state['candidates'] = {k: v for k, v in realtime_state['candidates'].items()
                                    if k in ('NOOP', 'RIGHT', 'RIGHT_A', 'RIGHT_B',
                                             'RIGHT_A_B', 'A', 'LEFT')}
    wire = attach_contract(realtime_state, profile)
    instructions = REFERENCE + (
        'Choose ONE complete candidate button combination from `candidates` now. '
        'This is a real-time controller: the emulator continues at 60fps while this '
        'request is in flight, and the selected buttons are held until the next '
        'Choice response arrives. Treat the supplied state as a timestamped snapshot; '
        'the answer can be stale by several frames. Do not invent what happened after '
        'the snapshot and do not assume the response-time scene is unchanged. '
        'Use `canonical_system_state` as the primary perception source. It comes from '
        'emulator telemetry and NES RAM, not screenshot interpretation; RGB-derived '
        '`analysis` is only a secondary consistency check. If `canonical_system_state.terrain.gap_ahead` '
        'is true, choose a forward jump or keep A held; do not continue RIGHT across the gap. '
        'Compare candidates using exact buttons, the local `execute_frames` horizon, '
        'support consequence, confirmed gaps, enemies, ceilings, jump phase, and '
        'landing surfaces. If a reachable question block or revealed item can be '
        'collected safely, choose the candidate that collects it before ordinary progress. '
        'Next prefer a safe descending stomp on a Goomba-like enemy when the landing remains '
        'recoverable; never seek a kill by side contact. Preserve a recoverable continuation '
        'first, then make rightward progress or gain needed height. Do not wait merely because it '
        'moves less; waiting or braking needs a concrete safety or positioning '
        'purpose. Unknown geometry is not a pit and is not evidence of safety. '
        'Use `reaction_timing` as the primary timing evidence: if jump_or_hold_must_this_decision '
        'is true, start or keep holding a forward jump; do not release A merely because Mario is '
        'already airborne. `jump_must_start_this_decision` includes the measured API delay and '
        'Follow `reaction_timing.jump_control_phase`: in `grounded_ready`, A may start a new jump; '
        'in `airborne_rising`, A only continues the current jump; in `airborne_descending`, do not '
        'treat A as a new jump and prioritize a recoverable landing or enemy stomp. '
        'the lead time needed to start a jump. Record whether a jump is needed and '
        'the current danger level as auxiliary judgments, but let the declared Choice determine '
        'the buttons. If every option is poor, choose the most recoverable listed action. '
        'Return only the declared Choice.'
    )
    return {'model': model, 'state': wire, 'questions': {
        'action': {'type': 'choice', 'instructions': instructions,
                   'criteria': {k: {'buttons': c['buttons'],
                                    'meaning': c['meaning'],
                                    'jump_effect': c['jump_effect'],
                                    'frames': c['execute_frames']}
                                for k, c in realtime_state['candidates'].items()}},
        'jump_needed': {'type': 'noul',
                        'instructions': 'Is a jump needed during this decision horizon to preserve a recoverable route? '
                                        'Use reaction_timing, confirmed gaps, enemy contact timing and headroom. '
                                        'If jump_or_hold_must_this_decision is true, answer true even when A is already held. '
                                        'This is auxiliary telemetry; it does not override action.',
                        'criteria': {'true': 'A jump should start or remain held during this horizon.',
                                     'false': 'A jump is unavailable, unnecessary, or unsafe during this horizon.'}},
        'danger': {'type': 'score',
                   'instructions': 'Score immediate danger to Mario during the response-delay horizon. '
                                   'Use 0 for clear recoverable space and 1 for imminent death or unrecoverable fall.',
                   'criteria': ['Safe open movement',
                                'Potential obstacle or enemy soon',
                                'Immediate collision, fall, or enemy threat']},
    }}


def assessments_from(body, state):
    expected = {'safe_'+k for k in state['candidates']}
    answers = body.get('answers', {})
    if set(answers) != expected:
        raise ValueError('Candidate assessment keys do not match; stopped without fallback.')
    result = {}
    for name in state['candidates']:
        answer = answers['safe_'+name]
        p = answer.get('noul')
        if (answer.get('type') != 'noul' or type(p) not in (int, float)
                or not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError('Invalid candidate Noul; stopped without fallback.')
        result[name] = {'survivable_continuation_probability': p}
    return result




def selected_action(body, state):
    answer = body.get('answers', {}).get('action', {})
    probs = answer.get('probabilities', {})
    if (answer.get('type') != 'choice' or answer.get('choice') not in state['candidates']
            or set(probs) != set(state['candidates'])
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in [*probs.values(), answer.get('confidence')])
            or abs(sum(probs.values())-1) > .02):
        raise ValueError('Invalid candidate Choice; stopped without fallback.')
    return answer['choice']
