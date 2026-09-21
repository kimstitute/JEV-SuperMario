"""Protect the experiment's observation and policy boundaries."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from jev_mario.policy import ACTIONS, validate_answer, request_for
from jev_mario.run import self_state, build_state, describe_observation, policy_state_for, observe_frame
from jev_mario.vision import ScreenObserver

FIXTURES = Path(__file__).parent/'fixtures'

def test_only_mario_state_is_exposed():
    info = dict(x_pos=226,left_x_pos=112,y_pixel=176,status='small',
                enemy_types=['SECRET'],hidden_map='SECRET',flag_get=False)
    ram = np.zeros(2048,dtype=np.uint8)
    mario = self_state(info,ram,None,0,[])
    assert 'SECRET' not in json.dumps(mario)
    assert mario['screen_box'] == [112,192,16,16]
    assert mario['vx'] is None  # not enough temporal evidence

def test_real_frame_recognizes_goomba_not_ground_or_mario():
    rgb = np.array(Image.open(FIXTURES/'enemy.png'))
    mario = {'world_x':226,'screen_x':112,'screen_box':[112,192,16,16]}
    obs,_ = ScreenObserver().detect(rgb,mario,120)
    assert len(obs['sprites']) == 1
    assert 90 < obs['sprites'][0]['dx'] < 115
    assert any(o['kind']=='question' for o in obs['terrain'])
    assert not obs['floor_gaps']

def test_camera_scroll_does_not_make_stationary_object_move():
    rgb = np.array(Image.open(FIXTURES/'enemy.png'))
    observer = ScreenObserver()
    mario = {'world_x':226,'screen_x':112,'screen_box':[112,192,16,16]}
    first,_ = observer.detect(rgb,mario,0)
    # Move the entire image left 4px and camera right 4px.
    shifted = np.roll(rgb,-4,axis=1)
    shifted[:,-4:] = [104,136,252]
    second,_ = observer.detect(shifted,{**mario,'world_x':230},4)
    assert first['sprites'][0]['id'] == second['sprites'][0]['id']
    assert second['sprites'][0]['vx'] == 0

def test_history_is_actual_chronological_frames_not_duplicate_padding():
    rgb = np.array(Image.open(FIXTURES/'initial.png'))
    observer = ScreenObserver()
    def sample(n):
        return observe_frame(observer,rgb,{
            'world_x':40,'screen_x':40,'screen_box':[40,192,16,16]},n)
    state,_ = build_state([sample(n) for n in range(13)],6)
    assert [o['frame'] for o in state['history']] == [0,4,8,12]
    assert [o['age_frames'] for o in state['history']] == [12,8,4,0]
    assert state['history'][-1]['observer_updates']==13
    initial,_ = build_state([observe_frame(ScreenObserver(),rgb,{'world_x':40,'screen_x':40,'screen_box':[40,192,16,16]},0)],6)
    assert len(initial['history']) == 1

def test_invalid_model_action_stops_instead_of_fallback():
    with pytest.raises(ValueError):
        validate_answer({'answers':{'action':{'type':'choice','choice':'TELEPORT'}}})
    probs = {name:1/len(ACTIONS) for name in ACTIONS}
    good = {'answers':{'action':{'type':'choice','choice':'RIGHT','probabilities':probs,'confidence':0.2}}}
    assert validate_answer(good) == 'RIGHT'
    good['answers']['action']['probabilities']['RIGHT'] = float('nan')
    with pytest.raises(ValueError):
        validate_answer(good)

def test_request_has_one_action_question_and_no_key():
    payload = request_for({'history':[]},'jev-1.13.0')
    assert list(payload['questions']) == ['action']
    assert set(payload['questions']['action']['criteria']) == set(ACTIONS)
    assert 'Authorization' not in json.dumps(payload)

def test_airborne_observation_keeps_ground_below_mario():
    obs = {'frame':20,'age_frames':0,'mario':{
        'screen_box':[112,120,16,16],'vx':3,'vy':4,'grounded':False,
        'form':'small','held_buttons':['right']},'sprites':[], 'floor_gaps':[],
        'terrain':[{'kind':'ground','dx':-120,'dy':72,'width':256,'height':16},
                   {'kind':'ground','dx':-120,'dy':88,'width':256,'height':16}]}
    texts = describe_observation(obs)['observations']
    floor = [s for s in texts if 'ground surface' in s]
    assert len(floor) == 1
    assert 'top 72px' in floor[0]
    assert 'falling' in texts[0]

def test_grid_keeps_layout_and_contains_no_prose_policy():
    rgb = np.array(Image.open(FIXTURES/'enemy.png'))
    sample = observe_frame(ScreenObserver(),rgb,{
        'world_x':226,'screen_x':112,'screen_box':[112,192,16,16]},0)
    raw,_ = build_state([sample],6)
    assert 'screen_grid' not in policy_state_for(raw,'raw')['history'][0]
    state = policy_state_for(raw,'grid')
    grid = state['history'][0]['screen_grid']
    assert len(grid)==25 and all(len(row)==32 for row in grid)
    assert grid[19][14:16] == 'MM'
    assert 'E' in grid[19][26:30]
    assert grid[21].count('#') >= 28
    assert 'sprites' not in state['history'][0]
    assert 'observations' not in state['history'][0]
