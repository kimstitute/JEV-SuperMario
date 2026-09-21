"""Direct Choice or parallel movement/start/hold judgments, no fallback."""
import math
import os
import time
import httpx
from .rulebook import DEFAULT_PROFILE, validate_profile, contract_metadata

ACTIONS = {
    'NOOP': [], 'RIGHT': ['right'], 'RIGHT_A': ['right', 'A'],
    'RIGHT_B': ['right', 'B'], 'RIGHT_A_B': ['right', 'A', 'B'],
    'A': ['A'], 'LEFT': ['left'],
    'LEFT_A': ['left','A'],
}
DESCRIPTIONS = {
    'NOOP': 'Release all buttons. Momentum and gravity continue.',
    'RIGHT': 'Hold right; release A and B.',
    'RIGHT_A': 'Hold right and jump button A; release B.',
    'RIGHT_B': 'Hold right and run button B; release A.',
    'RIGHT_A_B': 'Hold right, jump A, and run B together.',
    'A': 'Hold jump A; release directions and B.',
    'LEFT': 'Hold left; release A and B. This can decelerate rightward movement.',
    'LEFT_A': 'Hold left and A; release B.',
}
INSTRUCTIONS = (
    'Choose the button combination to hold for the next `action_frames` game frames '
    'using the chronological observations in `history`. Reach the right end of the '
    'level while avoiding death. The emulator is paused while you decide. '
    'Each option replaces all previously held buttons. A is the jump button; '
    'holding it during a jump affects jump height, and a new jump may require '
    'releasing and pressing it again. B runs. Mario retains momentum. '
    'Solid surfaces support Mario; touching an enemy from the side can kill him, '
    'while landing on a goomba from above can defeat it. '
    'Use canonical emulator telemetry/RAM state when supplied; its local collision grid and '
    'structured player/hazard/terrain fields are authoritative. RGB or screen-derived evidence '
    'is secondary and may be stale. Observe ceilings as well as ground. '
    'Unobserved space is unknown. Decide only from supplied observations and game rules. '
    'All observations use pixels, with x to the right and y down. '
    'When `grid_spec` is supplied, read each `screen_grid` as a two-dimensional '
    'screen using its legend, origin and cell size. The last history item is current. '
    'Use the separate terrain and knowledge layers to distinguish currently seen, '
    'remembered, visually empty, occluded and unknown regions. Unknown is not a pit. '
    'Remembered enemies have a last observed position, not a measured current position. '
    'Object dx/dy are relative to Mario feet center in that observation. '
    'Velocities are world pixel changes per game frame. '
    'No predicted trajectories, recommended actions, hidden map, or enemy RAM are supplied.'
)

MOVEMENTS = {
    'run_right':'Hold right+B for forward progress and horizontal jump distance.',
    'walk_right':'Hold right, release B, for finer positioning; existing momentum persists.',
    'brake_left':'Hold left to reduce rightward momentum or retreat from a hazard.',
    'wait':'Release horizontal buttons and B. Momentum persists; this is not an instant stop.',
}
COMMON = (
    'Control Super Mario Bros using `canonical_system_state` first, then `observation`, `analysis`, '
    '`history` and `recent_controls`. '
    'Finish the level without dying. The emulator waits for this response, then executes '
    '`action_frames` frames, with an early decision after landing. All questions see the same '
    'state independently; do not assume access to another answer. RGB evidence can be wrong. '
    'Rule signals and timing are approximate aids, not guaranteed safe actions. Unknown is '
    'not a pit; remembered objects are not current detections. A jumps; B runs. '
    'Jumping requires ground contact and a released A button. Holding A while rising '
    'extends a jump; pressing it again in midair cannot create another jump. '
)
COMPOSED_QUESTIONS = {
    'movement':{'type':'choice','instructions':COMMON + (
        'Choose horizontal movement for the next action interval. Jump is judged separately; '
        'do not wait simply because a forward obstacle needs jumping. Normally advance right. '
        'Use walk for precise takeoff or landing placement; use brake_left if moving forward '
        'would hit an enemy before a jump can clear it, or to rearm A with more space. '
        'Over a gap preserve enough forward motion to reach an observed landing surface. '
        'Inspect support heights: reaching the far wall below its top is not a safe landing. '
        'Repeated zero progress in recent_controls suggests the current approach is blocked.'),
        'criteria':MOVEMENTS},
    'start_jump':{'type':'noul','instructions':COMMON + (
        'Should Mario START a jump now? False if analysis.motion.can_start_jump is false. '
        'Use nearest enemy edge gap, closing speed and takeoff window, obstacle height, '
        'confirmed gaps, landing surfaces and ceiling clearance. An open enemy takeoff '
        'window is evidence for jumping now; a late window is urgent but may already be '
        'unrecoverable. With low ceiling, use the short-hop window rather than starting a '
        'long jump too early and landing into the enemy. A blocked nearby pipe or step '
        'with room overhead is a reason to jump even at zero forward speed. For stairs '
        'followed by a pit, land on a suitable step before launching across the pit. '
        'False on clear ground with no approaching hazard. Judge whether starting A NOW '
        'is useful; do not automatically accept a rule signal if the geometry contradicts it.'),
        'criteria':{'true':'Begin a jump during this interval to clear an approaching hazard or climb to a support.',
                    'false':'Do not start a jump now: unavailable, premature, unhelpful, or insufficient clearance.'}},
    'hold_jump':{'type':'noul','instructions':COMMON + (
        'Should Mario KEEP holding A during the CURRENT jump? False if grounded, A is '
        'already released, or descending. True while rising if more height or distance '
        'is needed to get over an obstacle or reach the top of a landing surface. '
        'A low-ceiling short hop may need release early. Release after a ceiling hit '
        'or when enough clearance is achieved, so the next jump can be armed. '
        'Holding A cannot push through a ceiling.'),
        'criteria':{'true':'Continue the currently held jump to gain useful height/distance.',
                    'false':'Release A to shorten/end this hold or rearm the next jump.'}},
}
NOUL_THRESHOLD=0.5

def request_for(state, model, controller='direct'):
    if controller=='composed':
        return {'model':model,'state':state,'questions':COMPOSED_QUESTIONS}
    return {'model': model, 'state': state, 'questions': {
        'action': {'type': 'choice', 'instructions': INSTRUCTIONS,
                   'criteria': DESCRIPTIONS}}}

def validate_answer(body):
    answer = body.get('answers', {}).get('action', {})
    action = answer.get('choice')
    probs = answer.get('probabilities', {})
    if answer.get('type') != 'choice' or action not in ACTIONS:
        raise ValueError('Jev returned an invalid action; execution stopped.')
    if set(probs) != set(ACTIONS):
        raise ValueError('Jev probability keys do not match the action set.')
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
           for v in probs.values()) or abs(sum(probs.values()) - 1) > 0.02:
        raise ValueError('Jev returned invalid probabilities.')
    confidence = answer.get('confidence')
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Jev returned invalid confidence.')
    return action

def _probability(value):
    return type(value) in (int,float) and math.isfinite(value) and 0<=value<=1

def compose_answers(body, state):
    """Validate ALL answers, then compose using declared button mechanics only.

    Hazards/rule signals never select or override a button in this function.
    The grounded/A-held gates prevent interpreting a start answer as a hold.
    """
    answers=body.get('answers',{})
    movement=answers.get('movement',{})
    probabilities=movement.get('probabilities',{})
    if (movement.get('type')!='choice' or movement.get('choice') not in MOVEMENTS
        or set(probabilities)!=set(MOVEMENTS)
        or not all(_probability(v) for v in probabilities.values())
        or abs(sum(probabilities.values())-1)>.02
        or not _probability(movement.get('confidence'))):
        raise ValueError('Invalid Jev movement answer; stopped without fallback.')
    for name in ('start_jump','hold_jump'):
        answer=answers.get(name,{})
        if answer.get('type')!='noul' or not _probability(answer.get('noul')):
            raise ValueError(f'Invalid Jev {name} answer; stopped without fallback.')
    mario=state['observation']['mario']
    grounded=mario['grounded']
    held='A' in mario['held_buttons']
    question='start_jump' if grounded else 'hold_jump'
    permitted=not held if grounded else held
    jump=permitted and answers[question]['noul']>=NOUL_THRESHOLD
    buttons={'run_right':['right','B'],'walk_right':['right'],
             'brake_left':['left'],'wait':[]}[movement['choice']]
    if jump:
        buttons=buttons+['A']
    action=next(name for name,value in ACTIONS.items() if set(value)==set(buttons))
    return {'action':action,'movement':movement['choice'],
            'start_jump_probability':answers['start_jump']['noul'],
            'hold_jump_probability':answers['hold_jump']['noul'],
            'active_jump_question':question,'jump_button_eligible':permitted,
            'jump_pressed':jump,'noul_threshold':NOUL_THRESHOLD,
            'mechanical_gate':'release_A_to_rearm' if grounded and held else 'A_already_released_in_air' if not grounded and not held else None,
            'buttons':list(ACTIONS[action])}

class TokenBudgetReached(RuntimeError):
    pass


class JevPolicy:
    def __init__(self, model='jev-1.13.0', timeout=40, controller='candidates', on_exchange=None,
                 prompt_profile=DEFAULT_PROFILE):
        validate_profile(prompt_profile)
        key = os.environ.get('TYPESAFE_API_KEY', '').strip()
        if not key:
            raise ValueError('Set TYPESAFE_API_KEY in .env first.')
        self.model = model
        self.controller = controller
        self.prompt_profile = prompt_profile
        self.on_exchange = on_exchange
        self.exchanges = []
        self.client = httpx.Client(
            base_url='https://api.typesafe.ai', timeout=timeout,
            headers={'Authorization': f'Bearer {key}'})

    def _post(self, payload, stage):
        exchange = {'stage': stage, 'request': payload}
        if self.controller == 'candidates':
            exchange['prompt_contract'] = contract_metadata(self.prompt_profile)
        if self.on_exchange:
            self.on_exchange('request', exchange)
        self.exchanges.append(exchange)
        started = time.perf_counter()
        response = self.client.post('/v1/systemone', json=payload)
        exchange['latency_s'] = round(time.perf_counter() - started, 4)
        exchange['http_status'] = response.status_code
        if response.status_code != 200:
            if self.on_exchange:
                self.on_exchange('error', exchange)
            # Do not log request headers or credentials, even on service errors.
            raise RuntimeError(f'TypeSafe HTTP {response.status_code}; run stopped (no fallback).')
        body = response.json()
        exchange['response'] = body
        if self.on_exchange:
            self.on_exchange('response', exchange)
        return body

    def choose(self, state):
        self.exchanges = []
        started = time.perf_counter()
        if self.controller == 'realtime':
            from .candidates import realtime_request, selected_action
            payload = realtime_request(state, self.model, self.prompt_profile)
            body = self._post(payload, 'realtime_action')
            # Validate against the exact candidate set sent on the wire. The
            # realtime controller deliberately uses the smaller seven-action
            # macro set even though the paused controller retains LEFT_A.
            action = selected_action(body, payload['state'])
            return action, payload, body, round(time.perf_counter() - started, 4)
        if self.controller == 'candidates':
            from .candidates import assessment_request, assessments_from, selection_request, selected_action
            assessment_payload = assessment_request(state, self.model, self.prompt_profile)
            assessment_body = self._post(assessment_payload, 'candidate_assessment')
            assessments = assessments_from(assessment_body, state)
            payload = selection_request(state, assessments, self.model, self.prompt_profile)
            body = self._post(payload, 'action_selection')
            action = selected_action(body, state)
        else:
            payload = request_for(state, self.model, self.controller)
            body = self._post(payload, self.controller)
            action = compose_answers(body,state)['action'] if self.controller=='composed' else validate_answer(body)
        return action, payload, body, round(time.perf_counter() - started, 4)

    def close(self):
        self.client.close()
