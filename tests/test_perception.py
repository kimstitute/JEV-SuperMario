"""Observable behavior regressions; synthetic scenes supplement real fixtures."""
from copy import deepcopy
import numpy as np
import pytest
from PIL import Image
from jev_mario.vision import ScreenObserver, ROOT, SKY
from jev_mario.run import observe_frame, build_state, policy_state_for

MARIO = {'world_x':40,'screen_x':40,'screen_box':[40,192,16,16]}

def canvas(color=SKY):
    return np.full((240,256,3),color,dtype=np.uint8)

def paste(rgb,name,x,y):
    art = Image.open(ROOT/'assets/templates'/f'{name}.png').convert('RGBA')
    im = Image.fromarray(rgb).convert('RGBA')
    im.alpha_composite(art,(x,y))
    return np.array(im.convert('RGB'))

@pytest.mark.parametrize('name,kind',[
    ('goomba_walk','goomba_candidate'),('goomba_walk_mirror','goomba_candidate'),
    ('used','used'),('stair','stair'),('pipe_top','pipe_top'),('pipe_body','pipe_body')])
def test_dictionary_ignores_background_and_recognizes_each_shape(name,kind):
    rgb = paste(canvas([123,55,211]),name,96,144)
    obs,_ = ScreenObserver().detect(rgb,MARIO,0)
    found = [o for o in obs['sprites']+obs['terrain'] if o['kind']==kind]
    assert len(found)==1
    assert found[0]['dx']==48 and found[0]['dy']==-64

@pytest.mark.parametrize('x',[-4,246])
def test_partially_offscreen_goomba_keeps_full_box_and_partial_evidence(x):
    obs,_ = ScreenObserver().detect(paste(canvas(),'goomba_walk',x,176),MARIO,0)
    assert len(obs['sprites'])==1
    obj=obs['sprites'][0]
    assert obj['dx']==x-48 and obj['width']==16
    assert 0.55 <= obj['visible_fraction'] < 1

def test_track_survives_animation_scroll_and_short_occlusion():
    observer=ScreenObserver()
    one,_=observer.detect(paste(canvas(),'goomba_walk',160,176),MARIO,0)
    two,_=observer.detect(paste(canvas(),'goomba_walk_mirror',155,176),{**MARIO,'world_x':44},1)
    assert two['sprites'][0]['id']==one['sprites'][0]['id']
    assert two['sprites'][0]['vx']==-1  # screen -5, camera +4
    three,_=observer.detect(canvas(),{**MARIO,'world_x':48},2)
    assert three['sprites'][0]['source']=='remembered'
    assert three['sprites'][0]['age_frames']==1
    four,_=observer.detect(paste(canvas(),'goomba_walk',145,176),{**MARIO,'world_x':52},3)
    assert four['sprites'][0]['id']==one['sprites'][0]['id']
    assert four['sprites'][0]['vx']==-1
    expired,_=observer.detect(canvas(),MARIO,16)
    assert not expired['sprites']

def test_terrain_is_kept_beneath_occluder_in_separate_layer():
    observer=ScreenObserver()
    rgb=paste(canvas(),'brick',96,144)
    original,_=observer.detect(rgb,MARIO,0)
    covered_mario={**MARIO,'screen_box':[96,144,16,16]}
    covered=rgb.copy()
    covered[144:160,96:112]=[248,56,0]
    hidden,_=observer.detect(covered,covered_mario,1)
    brick=next(o for o in hidden['terrain'] if o['kind']=='brick')
    assert brick['source']=='remembered' and brick['visibility']=='occluded'
    assert hidden['terrain_grid'][13][12:14]=='BB'
    assert hidden['knowledge_grid'][13][12:14]=='OO'
    assert hidden['entity_grid'][13][12:14]=='MM'
    revealed,_=observer.detect(rgb,MARIO,2)
    assert revealed['terrain'][0]['source']=='observed'
    assert original['terrain'][0]['last_seen_frame']==0  # no retroactive mutation

def test_block_replacement_and_positive_removal_evidence():
    observer=ScreenObserver()
    observer.detect(paste(canvas(),'question',96,144),MARIO,0)
    used,_=observer.detect(paste(canvas(),'used',96,144),MARIO,1)
    assert used['terrain_changes'][0]['from']=='question'
    assert used['terrain_changes'][0]['to']=='used'
    unknown=paste(canvas(),'used',96,144)
    unknown[144:160,96:112]=[100,100,100]
    for frame in range(2,6):
        obs,_=observer.detect(unknown,MARIO,frame)
        assert obs['terrain'][0]['source']=='remembered'
        assert not obs['terrain_changes']
    for frame in (6,7):
        obs,_=observer.detect(canvas(),MARIO,frame)
        assert obs['terrain']
    removed,_=observer.detect(canvas(),MARIO,8)
    assert not removed['terrain']
    assert removed['terrain_changes'][0]['to']=='visually_empty'

def test_nonconsecutive_frames_do_not_confirm_removal():
    observer=ScreenObserver()
    observer.detect(paste(canvas(),'brick',96,144),MARIO,0)
    for frame in (4,8,12):
        obs,_=observer.detect(canvas(),MARIO,frame)
        assert obs['terrain']

def floor_scene():
    rgb=canvas()
    for y in (208,224):
        for x in range(0,256,16):
            rgb=paste(rgb,'ground',x,y)
    return rgb

def test_unknown_is_not_empty_or_a_pit_and_true_opening_has_evidence():
    rgb=floor_scene()
    rgb[208:240,128:160]=[117,77,97]
    unknown,_=ScreenObserver().detect(rgb,MARIO,0)
    assert not unknown['floor_gaps']
    assert unknown['knowledge_grid'][22][16:20]=='XXXX'
    rgb[208:240,128:160]=SKY
    clear,_=ScreenObserver().detect(rgb,MARIO,0)
    assert len(clear['floor_gaps'])==1
    assert clear['floor_gaps'][0]['dx_start']==80
    assert clear['knowledge_grid'][22][16:20]=='....'
    # A newly missing remembered floor needs repeated positive confirmation.
    observer=ScreenObserver()
    observer.detect(floor_scene(),MARIO,0)
    first,_=observer.detect(rgb,MARIO,1)
    assert not first['floor_gaps']
    observer.detect(rgb,MARIO,2)
    confirmed,_=observer.detect(rgb,MARIO,3)
    assert confirmed['floor_gaps']

def test_every_frame_snapshot_survives_across_decision_boundaries():
    observer=ScreenObserver()
    samples=[]
    for frame in range(19):
        samples.append(observe_frame(observer,paste(canvas(),'goomba_walk',160-frame,176),MARIO,frame))
        if frame in (6,12,18):
            state,_=build_state(samples[-13:],6)
            latest=state['history'][-1]
            assert latest['observer_updates']==frame+1
            assert latest['sprites'][0]['id']=='v1'
            assert latest['sprites'][0]['vx']==-1
    frozen=deepcopy(samples[0])
    build_state(samples[-13:],6)
    assert samples[0]['observation']==frozen['observation']
    grid=policy_state_for(state,'grid')['history'][-1]
    assert all(key in grid for key in ('terrain_grid','entity_grid','knowledge_grid','tracks'))

def test_reset_requires_a_new_observer():
    observer=ScreenObserver()
    observer.detect(canvas(),MARIO,4)
    with pytest.raises(ValueError):
        observer.detect(canvas(),MARIO,0)
