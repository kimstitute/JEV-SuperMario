"""Shared request-visible rules, kept separate from measured facts and forecasts.

Manual: https://www.nintendo.co.jp/clv/manuals/en/pdf/CLV-P-NAAAE.pdf
Printed pp. 3-4: controls; pp. 8-10: damage, enemies and lethal gaps.
Collision geometry is a qualitative model; controller restrictions are named as such.
"""
from copy import deepcopy
import hashlib
import json

PROFILES = ('legacy-v2', 'rules-only-v1', 'rules-v1')
DEFAULT_PROFILE = 'rules-v1'
CONTRACT_KEYS = ('game_rules', 'observation_rules', 'control_contract', 'decision_policy')

GAME_RULES = {
    'version': 'smb1-ground-v1',
    'scope': 'NES Super Mario Bros 1-1 ground play. Do not assume unseen enemies, items or invincibility.',
    'controls': {
        'A': 'Jump. Holding A longer during ascent allows a higher jump than a short press.',
        'B': 'With a direction, run faster. Speed before takeoff affects the jump. Small Mario cannot shoot.',
        'momentum': 'Releasing directions is not an instant stop. Opposite input brakes before reversing.',
    },
    'contact': {
        'enemy_side': 'Enemy contact from the side or below harms Mario; it kills small Mario. '
                      'Powered Mario normally loses his power instead. Do not assume invincibility.',
        'goomba_top': 'Descending onto the top of a Goomba can defeat it. Side contact is not a stomp. '
                      'A goomba-like detection is uncertain; do not generalize stomping to all enemies.',
        'solid_side': 'Ground, pipes, stairs and blocks are solid. Their sides block movement; '
                      'touching a normal solid side is not itself enemy damage.',
        'solid_top': 'A top surface can support Mario when his feet reach it from above. '
                     'Reaching its side below the top is not a landing.',
        'ceiling': 'A solid underside can stop ascent. It is a head bump, not necessarily death. '
                   'An elevated surface ahead may be a step to climb, not a ceiling directly overhead.',
        'pit': 'Falling into a pit is lethal, including for powered Mario. Horizontal progress below '
               'the far bank does not mean a successful crossing.',
    },
    'items_and_enemies': {
        'question_blocks': 'A reachable question block may contain a coin or power-up, but its hidden '
                           'contents are unknown unless visibly revealed. If Mario can safely hit it '
                           'from below without losing a landing or entering danger, prefer collecting '
                           'the revealed item before leaving the area.',
        'enemy_goal': 'When a Goomba-like enemy can be stomped safely from above with a recoverable '
                      'landing, prefer the stomp before continuing right. Never trade a likely death '
                      'or unrecoverable pit fall for a kill.',
        'priority': 'Subject to survival and recoverability: safe reachable item collection, then safe '
                    'enemy defeat, then rightward progress. Do not wait indefinitely for an item or '
                    'enemy that is not reachable from the current state.',
    },
}

OBSERVATION_RULES = {
    'source_priority': 'For realtime decisions, `canonical_system_state` is the authoritative perception. '
                       'It is parsed from emulator info and NES RAM. RGB-derived `analysis` and raw '
                       'detections are secondary consistency/debug evidence, not the primary map.',
    'coordinates': 'System state uses world pixels and tile-local collision-grid coordinates. '
                   'RGB analysis, when present, uses pixels with x right and y down.',
    'motion': 'vx/vy are measured world pixels per game frame. Negative vy is rising. '
              'A zero or noisy vy alone does not prove grounded; use observation.mario.grounded.',
    'freshness': 'RAM state is current emulator telemetry. RGB-tracked objects may be remembered and '
                 'stale; use their age and timing validity. A remembered overlapping enemy remains an '
                 'immediate hazard candidate even when its velocity is uncertain.',
    'unknown': 'Missing detections and null values mean unknown, not zero distance or safe empty space. '
               'A gap requires the supplied confirmed-gap evidence. Unknown is also not certain danger.',
    'estimates': 'RAM-derived facts and local collision grid are canonical observations; projected '
                 'hazards, reaction timing, analysis, rules, forecast_rules and candidate forecasts are fallible computed aids. '
                 'They are not engine physics or safety guarantees. Check geometry and assumptions. '
                 'A past jump is an outcome under its own speed, button hold and contacts.',
}

CONTROL_CONTRACT = {
    'buttons': 'Each Choice is one listed complete button combination and replaces ALL held buttons. '
               'An omitted A is released. Choose a listed candidate only; code executes the final '
               'Choice without a hazard override.',
    'jump_start': 'This controller offers a jump start only while grounded with A previously released. '
                  'On landing with A held, its next decision releases A before any new jump.',
    'jump_hold': 'Holding A while airborne continues the current jump, not a second jump. '
                 'The controller offers no A re-press after an airborne release. Releasing early '
                 'therefore sacrifices the remaining opportunity to extend that jump.',
    'perception': 'The realtime model input is structured emulator/RAM state: player, trajectory, '
                  'hazard, terrain, reaction_timing, recent_control and episode. The local grid is '
                  'collision geometry, not a screenshot. RGB analysis is retained for diagnostics only.',
    'clock': 'The paused candidates controller advances game time only during execution and makes a '
             'new decision after execute_frames or earlier on landing. The realtime controller '
             'advances one frame at a time while a Choice request is in flight and applies its '
             'response from the next available frame. There is no hidden reflex controller.',
}

DECISION_POLICY = {
    'version': 'bounded-continuation-v1',
    'assessment_horizon_frames': 48,
    'assessment': 'Judge the specified interval and whether it preserves a feasible continuation '
                  'through the next local contact, takeoff or landing, at most 48 game frames ahead. '
                  'Future decisions may change buttons, but cannot undo a missed takeoff or released '
                  'jump hold. Do not invent unobserved platforms or a perfect recovery. Unknown '
                  'outcomes remain uncertain. This is not whole-level survival probability.',
    'selection_priority': [
        'Avoid evident death or loss of a recoverable path; a harmless wall stop is not death.',
        'Among safe reachable alternatives, collect a question-block item or revealed item before '
        'leaving the area when doing so does not lose the route or landing.',
        'After item opportunities, defeat a Goomba-like enemy with a safe descending stomp when the '
        'landing remains recoverable. Side contact is never an acceptable way to seek a kill.',
        'After those safe opportunities, make rightward progress or gain the height needed to pass the obstacle.',
        'Use braking or waiting for a concrete positioning/recovery purpose, not indefinite avoidance. '
        'If danger is comparable on clear ground, prefer useful progress; slower is not inherently safer.',
        'If no good option exists, choose the most recoverable listed option. Never fabricate safety.',
    ],
    'probabilities': 'Candidate Noul values are fallible judgments under this horizon, not guarantees '
                     'or an automatic threshold. Reconsider them against the geometry. Choice confidence '
                     'is concentration among alternatives, not the probability that Mario survives.',
}

REFERENCE = ('Apply `game_rules`, `observation_rules` and `control_contract` in the state. '
             'Keep these rules distinct from approximate `rules` and `forecast_rules`. ')


def validate_profile(profile):
    if profile not in PROFILES:
        raise ValueError(f'Unknown prompt profile: {profile}')


def attach_contract(state, profile):
    validate_profile(profile)
    # No mutation of the observations reused by other profiles or later requests.
    result = deepcopy(state)
    for key in CONTRACT_KEYS:
        result.pop(key, None)
    if profile != 'legacy-v2':
        result.update(game_rules=deepcopy(GAME_RULES), observation_rules=deepcopy(OBSERVATION_RULES),
                      control_contract=deepcopy(CONTROL_CONTRACT))
        if result.get('execution_mode') == 'realtime_choice_one_call':
            result['control_contract']['buttons'] = (
                'The selected button combination is applied from the next available frame and '
                'remains held while the next Choice request is in flight. It is not guaranteed to '
                'last exactly candidate.execute_frames; that field is the local evaluation horizon.')
            result['control_contract']['clock'] = (
                'Game time advances continuously at the emulator frame rate during the request. '
                'The response is based on a snapshot and may be stale by several frames; never '
                'assume the response-time scene is identical to the request-time scene.')
    if profile == 'rules-v1':
        result['decision_policy'] = deepcopy(DECISION_POLICY)
    return result


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    ensure_ascii=False).encode('utf-8')).hexdigest()


def contract_metadata(profile):
    state = attach_contract({},profile)
    return {'profile':profile,'contract_sha256':canonical_hash(state),
            'game_rules_version':state.get('game_rules',{}).get('version'),
            'decision_policy_version':state.get('decision_policy',{}).get('version')}
