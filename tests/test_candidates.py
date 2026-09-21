"""Support deadlines, action coupling and two-stage API failure boundaries."""
from copy import deepcopy
import httpx
import pytest
from test_analysis_policy import scene
from jev_mario.candidates import (candidate_state, assessment_request, assessments_from,
                                  selection_request, selected_action)
from jev_mario.policy import ACTIONS, JevPolicy, TokenBudgetReached


def raw(gap=None):
    obs = scene()
    obs['sprites'] = []
    if gap is not None:
        obs['floor_gaps'] = [{'dx_start':gap, 'dx_end':gap+32, 'surface_dy':0,
                              'evidence':'clear sky to bottom, observed ground at both edges'}]
        obs['terrain'] = [
            {'kind':'ground','dx':-120,'dy':0,'width':120+gap,'height':16,'source':'observed'},
            {'kind':'ground','dx':gap+32,'dy':0,'width':80,'height':16,'source':'observed'}]
    return {'goal':'finish', 'coordinates':'pixels', 'history':[obs],
            'recent_terrain_changes':[], 'action_frames':6}


def assessment(state, p=.7):
    return {'answers':{'safe_'+k:{'type':'noul','noul':p} for k in state['candidates']},
            'usage':{'input_tokens':100,'output_tokens':20}}


def choice(state, action):
    return {'answers':{'action':{'type':'choice','choice':action,'confidence':1.,
             'probabilities':{k:float(k==action) for k in state['candidates']}}},
            'usage':{'input_tokens':120,'output_tokens':30}}


def test_support_deadline_is_not_sprite_front_distance_and_horizon_reduces():
    state = candidate_state(raw(12))
    risk = state['analysis']['support_loss']
    assert risk['front_edge_distance_px'] == 4
    assert risk['frames_until_feet_center_over_gap_at_current_speed'] == 4
    assert risk['possible_support_loss_frames_at_current_speed'] == [.67, 7.33]
    assert risk['minimum_forward_travel_to_overlap_far_bank_px'] == 36
    assert state['action_frames'] == 1
    assert state['candidates']['RIGHT_A_B']['jump_effect'] == 'start'
    assert state['candidates']['LEFT']['approximate_braking_distance_px_range'] == [15,56.25]
    assert set(state['candidates']) == set(ACTIONS)  # unsafe choices are NOT filtered out


def test_missing_geometry_does_not_fabricate_gap_and_input_is_not_mutated():
    original = raw()
    saved = deepcopy(original)
    state = candidate_state(original)
    assert state['action_frames'] == 6
    assert state['analysis']['support_loss'] is None
    assert state['candidates']['NOOP']['support_consequence'] == 'unknown'
    assert original == saved


def test_coasting_keeps_momentum_and_braking_differs_from_running():
    state = candidate_state(raw())
    coast = state['candidates']['NOOP']['free_space_horizontal_travel_px_range']
    brake = state['candidates']['LEFT']['free_space_horizontal_travel_px_range']
    run = state['candidates']['RIGHT_B']['free_space_horizontal_travel_px_range']
    assert 0 < brake[0] < coast[0] < run[0]
    assert run == [18.,18.]


def test_airborne_hold_and_release_have_distinct_irreversible_consequences():
    original = raw()
    m = original['history'][0]['mario']
    m.update(grounded=False,held_buttons=['right','A'],vy=-5)
    original['history'][0]['terrain'].append(
        {'kind':'pipe_body','dx':24,'dy':-12,'width':32,'height':16,'source':'observed'})
    state = candidate_state(original)
    assert state['candidates']['RIGHT_A']['jump_effect'] == 'hold'
    assert 'NOT a second jump' in state['candidates']['RIGHT_A']['meaning']
    assert 'cannot restore' in state['candidates']['RIGHT']['meaning']
    assert state['candidates']['RIGHT']['remaining_rise_to_clear_nearest_obstacle_px'] == 12


def test_completed_jump_ignores_cached_samples_before_takeoff():
    from jev_mario.candidates import _recent_jumps
    obs = raw()['history'][0]
    obs['frame'] = 12
    stale = deepcopy(obs)
    stale.update(frame=8)
    stale['mario']['feet_y'] = 100  # unrelated older motion
    r = {'frame':12,'action':'RIGHT_A','raw_observations':{'history':[stale,obs]},
         'outcome':{'executed_frames':2,'mario_after':{'world_x':116,'feet_y':198,'grounded':False}}}
    next_obs = deepcopy(obs)
    next_obs['frame'] = 14
    next_obs['mario'].update(world_x=116,feet_y=198,grounded=False)
    landed = {'frame':14,'action':'RIGHT','raw_observations':{'history':[next_obs]},
              'outcome':{'executed_frames':6,'mario_after':{'world_x':126,'feet_y':208,'grounded':True}}}
    evidence = _recent_jumps([r,landed])
    assert evidence[0]['observed_rise_px'] == 10
    assert evidence[0]['initial_A_hold_frames'] == 2


@pytest.mark.parametrize('grounded,held', [(True,['A']), (False,[])])
def test_only_mechanically_ineffective_jump_duplicates_are_removed(grounded, held):
    original = raw()
    original['history'][0]['mario'].update(grounded=grounded,held_buttons=held)
    state = candidate_state(original)
    assert all('A' not in c['buttons'] for c in state['candidates'].values())
    assert 'LEFT' in state['candidates'] and 'RIGHT_B' in state['candidates']


def test_questions_name_candidate_and_second_request_contains_actual_answers():
    state = candidate_state(raw(12))
    req = assessment_request(state,'test')
    for k in state['candidates']:
        assert 'candidates.'+k in req['questions']['safe_'+k]['instructions']
    values = assessments_from(assessment(state),state)
    selection = selection_request(state,values,'test')
    assert selection['state']['jev_candidate_assessments'] == values
    # No safety threshold overrides the final Jev choice, even with urgent geometry.
    assert selected_action(choice(state,'RIGHT_B'),state) == 'RIGHT_B'


@pytest.mark.parametrize('bad', [None,True,'0.9',float('nan'),-1.,1.1])
def test_invalid_assessment_rejected(bad):
    state = candidate_state(raw())
    with pytest.raises(ValueError):
        assessments_from(assessment(state,bad),state)


def test_invalid_candidate_or_partial_assessment_rejected():
    state = candidate_state(raw())
    body = assessment(state)
    del body['answers']['safe_NOOP']
    with pytest.raises(ValueError):
        assessments_from(body,state)
    body = choice(state,'RIGHT_B')
    body['answers']['action']['probabilities']['TELEPORT'] = 0
    with pytest.raises(ValueError):
        selected_action(body,state)


def test_two_stage_policy_and_budget_boundary(monkeypatch):
    monkeypatch.setenv('TYPESAFE_API_KEY','unit-test-placeholder')
    state = candidate_state(raw(12))
    requests, events = [], []
    def handler(request):
        import json
        payload = json.loads(request.content)
        requests.append(payload)
        body = assessment(state) if len(requests)==1 else choice(state,'RIGHT_A_B')
        return httpx.Response(200,json=body)
    policy = JevPolicy(on_exchange=lambda event,x:events.append((event,x['stage'])))
    policy.client.close()
    policy.client = httpx.Client(transport=httpx.MockTransport(handler),base_url='https://example.test')
    try:
        action, payload, body, latency = policy.choose(state)
        assert action == 'RIGHT_A_B' and len(requests) == 2
        assert requests[1]['state']['jev_candidate_assessments']['RIGHT_B']['survivable_continuation_probability'] == .7
        assert len(policy.exchanges) == 2
        assert events == [('request','candidate_assessment'),('response','candidate_assessment'),
                          ('request','action_selection'),('response','action_selection')]
        requests.clear()
        def budget(event,x):
            if event=='request' and x['stage']=='action_selection':
                raise TokenBudgetReached()
        policy.on_exchange = budget
        with pytest.raises(TokenBudgetReached):
            policy.choose(state)
        assert len(requests) == 1
    finally:
        policy.close()


def test_failed_first_stage_never_calls_second_stage(monkeypatch):
    monkeypatch.setenv('TYPESAFE_API_KEY','unit-test-placeholder')
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(503)
    policy=JevPolicy()
    policy.client.close()
    policy.client=httpx.Client(transport=httpx.MockTransport(handler),base_url='https://example.test')
    try:
        with pytest.raises(RuntimeError,match='HTTP 503'):
            policy.choose(candidate_state(raw()))
        assert len(requests)==1
    finally:
        policy.close()
