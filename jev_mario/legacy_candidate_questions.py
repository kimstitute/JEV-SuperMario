# Frozen candidate-v2 questions for controlled comparisons. Do not edit.

def assessment_request(state, model):
    questions = {}
    for name in state['candidates']:
        questions['safe_' + name] = {
            'type': 'noul',
            'instructions': (
                f'Consider ONLY candidate `{name}` in `candidates.{name}`. Will executing '
                'these exact buttons for its execute_frames leave Mario alive with a '
                'plausible continuation to a supporting surface? Judge this candidate, '
                'not some different future action. For a jump, consider the required '
                'crossing distance, overhead obstacles, and far-bank height beyond this '
                'short interval. For continued running, consider whether waiting loses '
                'the opportunity to jump or brake. Ground support can be lost before '
                'another decision. A button release does not stop momentum. Forecasts '
                'include the consequence of releasing A while rising: this shortens the '
                'jump and cannot be undone until landing. Holding A during ascent means '
                'continuing the current jump, not initiating a forbidden second jump. '
                'Use remaining_rise_to_clear_nearest_obstacle_px and recent_completed_jumps '
                'to assess whether a short press has provided enough clearance. Forecasts '
                'are approximate; missing geometry is unknown. No question sees another answer.'),
            'criteria': {
                'true': 'This action preserves a plausible survivable continuation.',
                'false': 'This action risks collision, falling, or losing the last chance to recover.'}}
    return {'model': model, 'state': state, 'questions': questions}


def selection_request(state, assessments, model):
    return {'model': model, 'state': {**state, 'jev_candidate_assessments': assessments},
            'questions': {'action': {
                'type': 'choice',
                'instructions': (
                    'Choose ONE complete candidate button combination to execute now. '
                    'Reach the flag to the right while avoiding death. First consider '
                    '`analysis.support_loss`, enemies, obstacle height and landing surfaces; '
                    'then compare `candidates` with `jev_candidate_assessments` from an earlier '
                    'Jev request. Those probabilities are fallible evidence, not guarantees '
                    'or mandatory thresholds. Choose a coordinated horizontal+jump action. '
                    'Do not run off a confirmed gap while postponing the jump. A brief wait '
                    'or brake is useful only if it preserves a way forward. On clear ground '
                    'prefer progress over unnecessary jumps. During a jump keep sufficient '
                    'height and forward travel to reach the top of a landing surface. '
                    'While rising below an obstacle top, releasing A can make the jump '
                    'too short. Holding A continues the current jump; it does NOT start '
                    'another jump. Check the remaining rise and observed past jump heights. '
                    'A start press may need continued holding in subsequent decisions. '
                    'The emulator is paused for both requests; state has not changed. '
                    'If all options look poor, select the most recoverable available action.'),
                'criteria': {k: {'buttons': c['buttons'], 'meaning': c['meaning'],
                                  'jump_effect': c['jump_effect'], 'frames': c['execute_frames']}
                             for k, c in state['candidates'].items()}}}}
