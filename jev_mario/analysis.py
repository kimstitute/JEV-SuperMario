"""Deterministic geometry and timing analysis of RGB-derived observations.

No game RAM, model calls, emulator rollouts, or button selection here. Timing
estimates explicitly assume constant measured motion; jump windows are tunable
heuristics, not engine physics or guarantees of safety.
"""
from copy import deepcopy
import numpy as np

RULES = {
    'version':'rgb-analysis-v1',
    'normal_jump_clearance_frames':8,
    'low_ceiling_threshold_px':40,
    'minimum_hop_headroom_px':18,
    'low_ceiling_hop_edge_gap_px':[12,32],
    'obstacle_approach_px':48,
    'pit_approach_px':48,
    'notes':'Approximate heuristics for small Mario in SMB1. Timing uses current velocity, not a simulated trajectory. Unknown geometry is not empty. API wait advances zero game frames.',
}

def _horizontal_gap(obj, half):
    if obj['dx'] >= half:
        return 'ahead',obj['dx']-half
    if obj['dx']+obj['width'] <= -half:
        return 'behind',-half-obj['dx']-obj['width']
    return 'overlapping',0

def _surfaces(obs):
    """Expose only top edges, not subsurface rows or internal pipe seams."""
    m=obs['mario']
    ox=m['screen_box'][0]+m['screen_box'][2]/2
    oy=m['feet_y']
    solid=np.zeros((240,256),bool)
    observed=np.zeros((240,256),bool)
    for obj in obs['terrain']:
        x,y=round(ox+obj['dx']),round(oy+obj['dy'])
        l,t,r,b=max(x,0),max(y,40),min(x+obj['width'],256),min(y+obj['height'],240)
        if l>=r or t>=b:
            continue
        solid[t:b,l:r]=True
        if obj.get('source','observed')=='observed':
            observed[t:b,l:r]=True
    edges=solid.copy()
    edges[1:] &= ~solid[:-1]
    result=[]
    for y in np.where(edges.any(axis=1))[0]:
        for fresh in (True,False):
            row=edges[y] & (observed[y] if fresh else ~observed[y])
            changes=np.diff(np.pad(row.astype(int),(1,1)))
            for l,r in zip(np.where(changes==1)[0],np.where(changes==-1)[0]):
                if r-l<4:
                    continue
                result.append({'dx_start':float(l-ox),'dx_end':float(r-ox),'top_dy':float(y-oy),
                               'source':'observed' if fresh else 'remembered'})
    return sorted(result,key=lambda s:(s['dx_start'],s['top_dy']))

def analyze_observation(obs, action_frames):
    m=obs['mario']
    width,height=m['screen_box'][2:]
    half=width/2
    vx,vy=m['vx'],m['vy']
    grounded=m['grounded']
    phase='grounded' if grounded else 'rising' if vy is not None and vy<0 else 'falling' if vy is not None and vy>0 else 'airborne_unknown'
    enemies=[]
    for obj in obs['sprites']:
        relation,gap=_horizontal_gap(obj,half)
        fresh=obj.get('source','observed')=='observed'
        closing=None
        if fresh and vx is not None and obj.get('vx') is not None:
            closing=vx-obj['vx'] if relation=='ahead' else obj['vx']-vx if relation=='behind' else abs(vx-obj['vx'])
        overlap=max(-height,obj['dy']) < min(0,obj['dy']+obj['height'])
        # A remembered enemy is stale for trajectory estimation, but an
        # overlapping remembered box is still immediate hazard evidence. Keep
        # the timing marked uncertain while exposing the collision candidate.
        contact=0 if relation=='overlapping' and overlap else round(gap/closing,2) if closing is not None and closing>0 else None
        ground_level_contact = (contact if (grounded or (vy is not None and vy >= 0))
                                and abs(obj['dy']) <= max(height, 24) else None)
        enemies.append({**deepcopy(obj),'relation':relation,'edge_gap_px':gap,
                        'vertical_overlap_now':overlap,'closing_speed_px_per_frame':closing,
                        'horizontal_overlap_in_frames':contact,
                        'side_contact_in_frames_if_height_unchanged':contact if overlap else ground_level_contact,
                        'projected_ground_contact_in_frames':ground_level_contact,
                        'timing_valid':fresh and closing is not None})
    ahead=[e for e in enemies if e['relation'] in ('ahead','overlapping') and e.get('source','observed')=='observed']
    nearest=min(ahead,key=lambda e:e['edge_gap_px'],default=None)
    obstacles=[]
    ceilings=[]
    for obj in obs['terrain']:
        x,y,w,h=obj['dx'],obj['dy'],obj['width'],obj['height']
        if x+w<=-half:
            continue
        if y < 0 and y+h > -height and x>=half-2:
            top=y
            while True:
                above=[a['dy'] for a in obs['terrain']
                       if a['dy']+a['height']==top and max(a['dx'],x)<min(a['dx']+a['width'],x+w)]
                if not above:
                    break
                top=min(above)
            obstacles.append({**deepcopy(obj),'edge_gap_px':max(0,x-half),'height_above_feet_px':-top})
        if y+h<=-height and x<160:
            ceilings.append({'dx_start':x,'dx_end':x+w,'clearance_above_head_px':-height-y-h,
                             'source':obj.get('source','observed')})
    obstacle=min(obstacles,key=lambda o:o['edge_gap_px'],default=None)
    now=[c for c in ceilings if c['dx_start']<half and c['dx_end']>-half]
    headroom=min((c['clearance_above_head_px'] for c in now),default=None)
    corridor_end=nearest['dx']+nearest['width'] if nearest else 96
    corridor=[c for c in ceilings if c['dx_start']<=corridor_end and c['dx_end']>-half]
    forward_headroom=min((c['clearance_above_head_px'] for c in corridor),default=None)
    low=forward_headroom is not None and forward_headroom<=RULES['low_ceiling_threshold_px']
    held='A' in m['held_buttons']
    jump_available=grounded and not held
    enemy_window='not_applicable'
    if nearest and nearest['vertical_overlap_now'] and grounded:
        gap=nearest['edge_gap_px']
        contact=nearest['side_contact_in_frames_if_height_unchanged']
        if nearest['closing_speed_px_per_frame'] is not None and nearest['closing_speed_px_per_frame']<=0 and gap>0:
            enemy_window='not_closing'
        elif low:
            lo,hi=RULES['low_ceiling_hop_edge_gap_px']
            enemy_window='insufficient_headroom' if forward_headroom<RULES['minimum_hop_headroom_px'] else 'late' if gap<lo else 'open' if gap<=hi else 'approaching'
        elif contact is None:
            enemy_window='unknown_timing'
        else:
            clearance=RULES['normal_jump_clearance_frames']
            enemy_window='late' if contact<clearance else 'open' if contact<=clearance+action_frames else 'approaching'
    surfaces=_surfaces(obs)
    supports=[s for s in surfaces if s['dx_start']<half and s['dx_end']>-half and s['top_dy']>=-2]
    support=min(supports,key=lambda s:s['top_dy'],default=None)
    gaps=[{**deepcopy(g),'edge_gap_px':max(0,g['dx_start']-half),
           'visible_width_px':g['dx_end']-g['dx_start']} for g in obs['floor_gaps'] if g['dx_end']>-half]
    gap=min(gaps,key=lambda g:g['edge_gap_px'],default=None)
    knowledge=''.join(obs.get('knowledge_grid',[]))
    return {
        'motion':{'phase':phase,'vx_px_per_frame':vx,'vy_px_per_frame':vy,
                  'a_held':held,'can_start_jump':jump_available,
                  'must_release_a_before_next_jump':grounded and held},
        'enemies':enemies,'nearest_forward_enemy':nearest,
        'terrain':{'nearest_obstacle':obstacle,'nearest_confirmed_gap':gap,
                   'support_below':support,'landing_surfaces':surfaces,
                   'headroom_now_px':headroom,'headroom_before_enemy_px':forward_headroom,
                   'low_ceiling_before_enemy':bool(nearest and low),'ceilings':ceilings},
        'rule_signals':{
            'enemy_takeoff_window':enemy_window,
            'enemy_window_rule':'short_hop_edge_gap' if nearest and low else 'constant_speed_contact_minus_clearance',
            'obstacle_within_approach_range':bool(obstacle and obstacle['edge_gap_px']<=RULES['obstacle_approach_px']),
            'confirmed_gap_within_approach_range':bool(gap and gap['edge_gap_px']<=RULES['pit_approach_px']),
            'contact_within_next_action':bool(nearest and nearest['side_contact_in_frames_if_height_unchanged'] is not None and nearest['side_contact_in_frames_if_height_unchanged']<=action_frames),
        },
        'uncertainty':{'unknown_cells':knowledge.count('X'),'occluded_cells':knowledge.count('O'),
                       'remembered_terrain_cells':knowledge.count('R'),
                       'no_detected_enemy_does_not_mean_safe':True,
                       'no_detected_ceiling_does_not_mean_unlimited_headroom':True},
    }

def analyzed_state(state, records=()):
    current=state['history'][-1]
    analyses=[analyze_observation(o,state['action_frames']) for o in state['history']]
    recent=[]
    for record in records[-4:]:
        before=record['raw_observations']['history'][-1]['mario']
        recent.append({'frame':record['frame'],'buttons':record['action'],
                       'executed_frames':record['outcome']['executed_frames'],
                       'progress_px':record['outcome']['world_x']-before['world_x'],
                       'before':{k:before[k] for k in ('world_x','feet_y','vx','vy','grounded')},
                       'after':record['outcome'].get('mario_after')})
    return {
        'goal':state['goal'],'action_frames':state['action_frames'],
        'coordinates':state['coordinates'],'rules':deepcopy(RULES),
        'observation':{k:deepcopy(current[k]) for k in ('frame','mario','camera','terrain','sprites','floor_gaps','system_state') if k in current},
        'analysis':analyses[-1],
        'history':[{'frame':o['frame'],'age_frames':o['age_frames'],'mario':deepcopy(o['mario']),
                    'nearest_enemy':a['nearest_forward_enemy'],'jump_phase':a['motion']['phase'],
                    'headroom_px':a['terrain']['headroom_now_px']}
                   for o,a in zip(state['history'],analyses)],
        'recent_controls':recent,'recent_terrain_changes':state['recent_terrain_changes'],
        'observation_format':'RGB measurements plus deterministic geometry and approximate timing rules. No other AI and no enemy/map RAM.',
    }
