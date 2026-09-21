"""Physical geometry, uncertainty and Jev-to-buttons contract regressions."""
from copy import deepcopy
import math
import pytest
from jev_mario.analysis import analyze_observation, analyzed_state
from jev_mario.policy import MOVEMENTS, compose_answers, request_for

def scene():
    return {'frame':12,'age_frames':0,
            'mario':{'world_x':112,'screen_x':112,'screen_box':[112,192,16,16],
                     'feet_y':208,'vx':3.,'vy':0.,'grounded':True,'held_buttons':['right','B'],'form':'small'},
            'camera':{'world_x_offset':0},
            'terrain':[{'kind':'ground','dx':-120,'dy':0,'width':256,'height':16,'source':'observed'},
                       {'kind':'ground','dx':-120,'dy':16,'width':256,'height':16,'source':'observed'}],
            'sprites':[{'id':'v1','kind':'goomba_candidate','dx':38,'dy':-15,'width':16,'height':16,
                        'vx':-1.,'vy':0.,'source':'observed','age_frames':0}],
            'floor_gaps':[],'knowledge_grid':['X'*32 for _ in range(25)]}

def response(movement='run_right',start=.8,hold=.2):
    return {'answers':{
        'movement':{'type':'choice','choice':movement,'probabilities':{k:float(k==movement) for k in MOVEMENTS},'confidence':1.},
        'start_jump':{'type':'noul','noul':start},'hold_jump':{'type':'noul','noul':hold}}}

def test_edge_gap_and_relative_closing_speed_have_explicit_assumptions():
    a=analyze_observation(scene(),6)
    enemy=a['nearest_forward_enemy']
    assert enemy['edge_gap_px']==30
    assert enemy['closing_speed_px_per_frame']==4
    assert enemy['side_contact_in_frames_if_height_unchanged']==7.5
    assert a['rule_signals']['enemy_takeoff_window']=='late'
    assert a['terrain']['headroom_now_px'] is None
    assert a['terrain']['nearest_confirmed_gap'] is None
    assert a['uncertainty']['unknown_cells']==800

def test_low_ceiling_changes_timing_window_and_remains_measurement():
    obs=scene()
    obs['terrain'].append({'kind':'brick','dx':-8,'dy':-64,'width':80,'height':16,'source':'observed'})
    a=analyze_observation(obs,6)
    assert a['terrain']['headroom_now_px']==32
    assert a['terrain']['low_ceiling_before_enemy']
    assert a['rule_signals']['enemy_takeoff_window']=='open'
    obs['sprites'][0]['dx']=90
    assert analyze_observation(obs,6)['rule_signals']['enemy_takeoff_window']=='approaching'
    obs['terrain'][-1]['dy']=-40
    assert analyze_observation(obs,6)['rule_signals']['enemy_takeoff_window']=='insufficient_headroom'

def test_stale_enemy_cannot_create_current_contact_prediction():
    obs=scene()
    obs['sprites'][0].update(source='remembered',age_frames=8)
    a=analyze_observation(obs,6)
    assert a['nearest_forward_enemy'] is None
    assert a['enemies'][0]['horizontal_overlap_in_frames'] is None
    obs['sprites'][0].update(source='observed',vx=5.)
    assert analyze_observation(obs,6)['rule_signals']['enemy_takeoff_window']=='not_closing'

def test_pipe_height_includes_cap_and_landings_exclude_internal_seams():
    obs=scene()
    obs['terrain'] += [
        {'kind':'pipe_top','dx':40,'dy':-48,'width':32,'height':16,'source':'observed'},
        {'kind':'pipe_body','dx':40,'dy':-32,'width':32,'height':16,'source':'observed'},
        {'kind':'pipe_body','dx':40,'dy':-16,'width':32,'height':16,'source':'observed'}]
    a=analyze_observation(obs,6)
    assert a['terrain']['nearest_obstacle']['height_above_feet_px']==48
    tops={s['top_dy'] for s in a['terrain']['landing_surfaces']}
    assert tops=={-48.,0.}
    assert a['terrain']['support_below']['top_dy']==0

def test_start_hold_and_rearm_are_separate_mechanical_gates():
    state={'observation':scene()}
    assert compose_answers(response(),state)['action']=='RIGHT_A_B'
    state['observation']['mario']['held_buttons'].append('A')
    rearm=compose_answers(response(start=1.,hold=1.),state)
    assert rearm['action']=='RIGHT_B'
    assert rearm['mechanical_gate']=='release_A_to_rearm'
    state['observation']['mario']['grounded']=False
    assert compose_answers(response(start=0.,hold=1.),state)['action']=='RIGHT_A_B'
    assert compose_answers(response(start=1.,hold=0.),state)['action']=='RIGHT_B'
    state['observation']['mario']['held_buttons']=[]
    assert compose_answers(response(start=1.,hold=1.),state)['action']=='RIGHT_B'

@pytest.mark.parametrize('movement,action',[('run_right','RIGHT_A_B'),('walk_right','RIGHT_A'),('brake_left','LEFT_A'),('wait','A')])
def test_all_movement_and_jump_combinations_are_representable(movement,action):
    assert compose_answers(response(movement=movement),{'observation':scene()})['action']==action

def test_rules_never_override_a_valid_negative_jev_answer():
    obs=scene()
    obs['sprites'][0]['dx']=9
    state={'observation':obs,'analysis':analyze_observation(obs,6)}
    assert state['analysis']['rule_signals']['contact_within_next_action']
    result=compose_answers(response(start=0.,hold=0.),state)
    assert result['action']=='RIGHT_B'

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-0.1,1.1,True,'0.9',None])
def test_invalid_noul_stops_before_any_action(bad):
    body=response(start=bad)
    with pytest.raises(ValueError):
        compose_answers(body,{'observation':scene()})

def test_missing_even_inactive_answer_stops_and_no_fake_action_probabilities():
    body=response()
    del body['answers']['hold_jump']
    with pytest.raises(ValueError):
        compose_answers(body,{'observation':scene()})
    result=compose_answers(response(),{'observation':scene()})
    assert 'probabilities' not in result

def test_compact_analyzed_request_contains_evidence_and_three_parallel_questions():
    obs=scene()
    state={'history':[obs],'action_frames':6,'goal':'finish','coordinates':'pixels','recent_terrain_changes':[]}
    original=deepcopy(state)
    compact=analyzed_state(state)
    request=request_for(compact,'jev-1.13.0','composed')
    assert set(request['questions'])=={'movement','start_jump','hold_jump'}
    assert compact['observation']['sprites']==obs['sprites']
    assert 'screen_grid' not in compact['observation']
    assert compact['analysis']['nearest_forward_enemy']['edge_gap_px']==30
    assert state==original
