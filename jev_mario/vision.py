"""Persistent RGB perception for SMB1 outdoor scenes; no map/enemy RAM.

Appearance matching, camera coordinates, tracking and memory are measurements,
not a controller. Unsupported pixels remain unknown. Sky is visually empty,
not proof that the game contains no invisible collision object.
"""
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SKY = np.array([104,136,252], dtype=np.uint8)
SYMBOLS = {'ground':'#','brick':'B','question':'?','used':'U','stair':'S',
           'pipe_top':'P','pipe_body':'P'}

def components(mask, min_area=1):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(int(x),int(y),int(w),int(h),int(a))
            for x,y,w,h,a in stats[1:] if a >= min_area]

def contains_overlap(a, b):
    x,y,w,h = a
    u,v,s,t = b
    return max(x,u) < min(x+w,u+s) and max(y,v) < min(y+h,v+t)

def bounds(box):
    x,y,w,h = map(int,box)
    return max(0,x),max(40,y),min(256,x+w),min(240,y+h)

def mark(mask, box, value=True):
    x,y,r,b = bounds(box)
    if r > x and b > y:
        mask[y:b,x:r] = value

class ScreenObserver:
    def __init__(self, detection_interval=1):
        self.detection_interval = max(1, int(detection_interval))
        self.templates = []
        names = ['ground','brick','question','used','stair','pipe_top','pipe_body',
                 'goomba_walk','goomba_walk_mirror']
        for name in names:
            im = np.array(Image.open(ROOT/'assets/templates'/f'{name}.png').convert('RGBA'))
            mask = ((im[:,:,3] > 0) & np.any(im[:,:,:3] != SKY, axis=2)).astype(np.float32)
            self.templates.append({'name':name,'kind':'goomba_candidate' if name.startswith('goomba') else name,
                                   'rgb':im[:,:,:3].astype(np.float32),'mask':mask})
            if name in ('question','used'):
                for phase,color in enumerate(([228,92,16],[136,20,0]),1):
                    variant=im[:,:,:3].copy()
                    variant[np.all(variant==[252,160,68],axis=2)] = color
                    self.templates.append({'name':f'{name}_palette_{phase}','kind':name,
                                           'rgb':variant.astype(np.float32),'mask':mask})
        self.tracks = {}
        self.memory = {}
        self.next_id = 1
        self.last_frame = None
        self.last_camera = None
        self.updates = 0
        self.last_detection_frame = None
        self.cached_observation = None
        self.gap_counts = {}

    def _fast_observation(self, rgb, mario, frame_number):
        """Update tracked geometry between expensive template-match frames.

        The realtime controller still records a perception sample every frame;
        only full RGB appearance matching is decimated. World-relative motion
        and Mario state are refreshed so stale pixels are never presented as a
        newly observed detection.
        """
        if self.cached_observation is None:
            return None
        result = deepcopy(self.cached_observation)
        previous = result['mario']
        dt = max(1, frame_number - previous['frame'])
        world_delta = mario['world_x'] - previous['world_x']
        for obj in result.get('sprites', []):
            if obj.get('source') == 'observed' and obj.get('vx') is not None:
                obj['dx'] = round(obj['dx'] + (obj['vx'] - (mario.get('vx') or 0)) * dt, 1)
            else:
                obj['dx'] = round(obj['dx'] - world_delta, 1)
            obj['source'] = 'remembered'
            obj['age_frames'] = frame_number - obj.get('last_seen_frame', previous['frame'])
        for obj in result.get('terrain', []):
            obj['dx'] = round(obj['dx'] - world_delta, 1)
            if obj.get('source') == 'observed':
                obj['source'] = 'remembered'
            obj['age_frames'] = frame_number - obj.get('last_seen_frame', previous['frame'])
            obj['visibility'] = 'unconfirmed'
        result['frame'] = frame_number
        result['mario'] = deepcopy(mario)
        result['camera'] = {'world_x_offset': mario['world_x'] - mario['screen_x'],
                            'delta_x': None, 'source': 'Mario world_x minus screen_x'}
        result['floor_gaps'] = [{**g, 'dx_start': round(g['dx_start'] - world_delta, 1),
                                 'dx_end': round(g['dx_end'] - world_delta, 1)}
                                for g in result.get('floor_gaps', [])]
        result['observer_updates'] = self.updates + 1
        result['perception'] = result.get('perception', '') + ' Intermediate tracked frame; full RGB match not run.'
        self.updates += 1
        self.cached_observation = result
        return result

    def _matches(self, rgb, mario, dynamic=False, occluders=()):
        """Foreground SSD with explicit visibility support, including image edges.

        Padding carries zero visibility, never synthetic sky evidence. Mario's
        rectangle is also excluded. At least 55% of an appearance must be seen.
        """
        pad = 32
        image = np.pad(rgb.astype(np.float32),((pad,pad),(pad,pad),(0,0)))
        visible = np.zeros(image.shape[:2],np.float32)
        visible[pad+40:pad+240,pad:pad+256] = 1
        mx,my,mw,mh = mario['screen_box']
        for box in ([mx-1,my-1,mw+2,mh+2],*occluders):
            x,y,r,b = bounds(box)
            visible[pad+y:pad+b,pad+x:pad+r] = 0
        image *= visible[:,:,None]
        energy = (image*image).sum(axis=2)
        candidates = []
        for t in self.templates:
            if (t['kind']=='goomba_candidate') != dynamic:
                continue
            template,mask = t['rgb'],t['mask']
            h,w = mask.shape
            support = cv2.matchTemplate(visible,mask,cv2.TM_CCORR)
            error = cv2.matchTemplate(energy,mask,cv2.TM_CCORR)
            error += cv2.matchTemplate(visible,(template*template).sum(axis=2)*mask,cv2.TM_CCORR)
            error -= 2*cv2.matchTemplate(image,template*mask[:,:,None],cv2.TM_CCORR)
            mse = np.maximum(error,0)/np.maximum(support*3,1)
            valid = (support >= mask.sum()*0.55) & (mse < 100)
            ys,xs = np.where(valid)
            for py,px in zip(ys,xs):
                xx,yy = int(px-pad),int(py-pad)
                if yy+h <= 40 or yy >= 240 or xx+w <= 0 or xx >= 256:
                    continue
                candidates.append({'kind':t['kind'],'box':[xx,yy,w,h],
                                   'appearance':t['name'],'match_error':round(float(mse[py,px]),2),
                                   'visible_fraction':round(float(support[py,px]/mask.sum()),2)})
        result = []
        # Shafts repeat every scanline; use the native screen tile lattice.
        for obj in sorted(candidates,key=lambda o:(o['match_error'],o['box'][1],o['box'][0])):
            if obj['kind']=='pipe_body' and obj['box'][1] % 16:
                continue
            x,y,w,h = obj['box']
            if any(contains_overlap(obj['box'],old['box']) and
                   min(x+w,old['box'][0]+old['box'][2])-max(x,old['box'][0]) > min(w,old['box'][2])*0.4 and
                   min(y+h,old['box'][1]+old['box'][3])-max(y,old['box'][1]) > min(h,old['box'][3])*0.4
                   for old in result):
                continue
            result.append(obj)
        return result

    def _track(self, detections, camera, frame):
        available = set(self.tracks)
        current = []
        for obj in sorted(detections,key=lambda o:o['box'][0]):
            x,y,w,h = obj['box']
            wx = x+camera
            candidates = []
            for key in available:
                old = self.tracks[key]
                dt = frame-old['last_seen_frame']
                predicted = old['world_x']+(old['vx'] or 0)*dt
                distance = abs(wx-predicted)+abs(y-old['y'])
                if 0 < dt <= 12 and distance <= 8+3*dt:
                    candidates.append((distance,key))
            if candidates:
                _,key = min(candidates)
                available.remove(key)
                old = self.tracks[key]
                history = [p for p in old['positions'] if frame-p[0]<=4]
                if not history:
                    history = old['positions'][-1:]
                before,bx,by = history[0]
                dt = frame-before
                vx,vy = round((wx-bx)/dt,2),round((y-by)/dt,2)
            else:
                key = f'v{self.next_id}'
                self.next_id += 1
                vx=vy=None
                history=[]
            track = {**obj,'id':key,'world_x':wx,'y':y,'vx':vx,'vy':vy,
                     'last_seen_frame':frame,'age_frames':0,'source':'observed',
                     'positions':history+[(frame,wx,y)]}
            self.tracks[key] = track
            current.append(track)
        for key in available:
            old = self.tracks[key]
            age = frame-old['last_seen_frame']
            if age > 12:
                del self.tracks[key]
                continue
            box = [old['world_x']-camera,old['y'],*old['box'][2:]]
            if box[0]+box[2] > 0 and box[0] < 256:
                current.append({**old,'box':box,'source':'remembered','age_frames':age})
        return current

    def _terrain(self, detections, camera, frame, sky, occluded):
        changed = []
        seen = set()
        for obj in detections:
            x,y,w,h = obj['box']
            wx = x+camera
            key = next((key for key,old in self.memory.items()
                        if abs(key[0]-wx)<=2 and abs(key[1]-y)<=2 and old['box'][2:]==[w,h]),(wx,y))
            old = self.memory.get(key)
            if old and old['kind'] != obj['kind']:
                changed.append({'world_x':key[0],'y':y,'from':old['kind'],'to':obj['kind'],'evidence':'new appearance'})
            self.memory[key] = {**obj,'last_seen_frame':frame,'empty_streak':0}
            seen.add(key)
        visible = []
        for key,old in list(self.memory.items()):
            wx,y = key
            box = [wx-camera,y,*old['box'][2:]]
            x,top,r,b = bounds(box)
            if r<=x or b<=top:
                if abs(wx-camera)>1024:
                    del self.memory[key]
                continue
            covered = bool(occluded[top:b,x:r].any())
            clear = bool(sky[top:b,x:r].all()) and not covered
            fully_visible = x==box[0] and r==box[0]+box[2] and top==box[1] and b==box[1]+box[3]
            if key not in seen and clear and fully_visible:
                old['empty_streak'] = old['empty_streak']+1 if self.last_frame==frame-1 else 1
                if old['empty_streak'] >= 3:
                    changed.append({'world_x':wx,'y':y,'from':old['kind'],'to':'visually_empty','evidence':'3 consecutive clear-sky frames'})
                    del self.memory[key]
                    continue
            else:
                old['empty_streak'] = 0
            visible.append({**old,'box':box,'source':'observed' if key in seen else 'remembered',
                            'age_frames':frame-old['last_seen_frame'],
                            'visibility':'occluded' if covered else 'visible' if key in seen else 'unconfirmed'})
        return visible,changed

    def detect(self, rgb, mario, frame_number):
        if self.last_frame is not None and frame_number <= self.last_frame:
            raise ValueError('Observer frames must increase; create a new observer after reset.')
        if (self.detection_interval > 1 and self.last_detection_frame is not None
                and frame_number - self.last_detection_frame < self.detection_interval):
            result = self._fast_observation(rgb, mario, frame_number)
            if result is not None:
                self.last_camera = mario['world_x'] - mario['screen_x']
                self.last_frame = frame_number
                return result, []
        enemies = self._matches(rgb,mario,dynamic=True)
        matches = self._matches(rgb,mario,occluders=[o['box'] for o in enemies])
        camera = mario['world_x']-mario['screen_x']
        sprites = self._track(enemies,camera,frame_number)
        occluded = np.zeros((240,256),bool)
        mx,my,mw,mh = mario['screen_box']
        mark(occluded,[mx-1,my-1,mw+2,mh+2])
        for obj in sprites:
            if obj['source']=='observed':
                mark(occluded,obj['box'])
        sky = np.all(rgb==SKY,axis=2)
        terrain,changes = self._terrain([o for o in matches if o['kind']!='goomba_candidate'],
                                        camera,frame_number,sky,occluded)
        terrain_pixels = np.full((240,256),'.',dtype='<U1')
        freshness = np.full((240,256),'X',dtype='<U1')
        freshness[sky] = '.'
        for obj in terrain:
            mark(terrain_pixels,obj['box'],SYMBOLS[obj['kind']])
            mark(freshness,obj['box'],'T' if obj['source']=='observed' else 'R')
        freshness[occluded] = 'O'
        entity_pixels = np.full((240,256),'.',dtype='<U1')
        for obj in sprites:
            mark(entity_pixels,obj['box'],'E' if obj['source']=='observed' else 'e')
        mark(entity_pixels,mario['screen_box'],'M')
        grids = {k:[] for k in ('terrain_grid','entity_grid','knowledge_grid','screen_grid')}
        for row in range(25):
            lines = {k:[] for k in grids}
            for col in range(32):
                y,x = 40+row*8,col*8
                ty,tx = y+4,x+4
                ground = terrain_pixels[ty,tx]
                entity = entity_pixels[ty,tx]
                patch = freshness[y:y+8,x:x+8]
                knowledge = 'O' if (patch=='O').any() else 'T' if freshness[ty,tx]=='T' else 'R' if freshness[ty,tx]=='R' else '.' if (patch=='.').all() else 'X'
                lines['terrain_grid'].append(ground)
                lines['entity_grid'].append(entity)
                lines['knowledge_grid'].append(knowledge)
                lines['screen_grid'].append(entity if entity!='.' else ground if ground!='.' else knowledge)
            for key in grids:
                grids[key].append(''.join(lines[key]))
        gaps = []
        ground = [o for o in terrain if o['kind']=='ground' and o['source']=='observed']
        if ground:
            floor = min(o['box'][1] for o in ground)
            support = np.zeros(256,bool)
            for obj in ground:
                if obj['box'][1]==floor:
                    x,_,r,_ = bounds(obj['box'])
                    support[x:r] = True
            clear = sky[floor:240].all(axis=0) & ~occluded[floor:240].any(axis=0)
            clear &= (terrain_pixels[floor:240]=='.').all(axis=0)
            for x,_,w,_,_ in components(clear[None,:],8):
                if x>0 and x+w<256 and support[x-1] and support[x+w]:
                    gaps.append({'dx_start':x-(mario['screen_box'][0]+8),'dx_end':x+w-(mario['screen_box'][0]+8),
                                 'surface_dy':floor-sum(mario['screen_box'][1::2]),
                                 'kind':'visually_confirmed_floor_opening','evidence':'clear sky to bottom, observed ground at both edges'})
            # Pixel-clear tests can be invalidated by Mario/enemy occlusion or
            # a remembered terrain cell. The observed ground intervals are a
            # stronger geometric signal: a real horizontal separation between
            # two same-height support segments is a confirmed gap.
            segments = sorted((o['box'][0], o['box'][0] + o['box'][2])
                              for o in ground if o['box'][1] == floor)
            for (_, left_end), (right_start, _) in zip(segments, segments[1:]):
                gap_is_sky = bool(sky[floor:240, left_end:right_start].all())
                remembered_cover = any(o.get('kind') == 'ground' and o.get('source') == 'remembered'
                                       and o['box'][1] == floor
                                       and o['box'][0] <= left_end
                                       and o['box'][0] + o['box'][2] >= right_start
                                       for o in terrain)
                gap_key = (int(left_end), int(right_start), int(floor))
                self.gap_counts[gap_key] = self.gap_counts.get(gap_key, 0) + 1
                required_observations = 1 if self.last_frame is None else 3
                if (right_start - left_end >= 8 and gap_is_sky and not remembered_cover
                        and self.gap_counts[gap_key] >= required_observations):
                    candidate = {'dx_start':left_end-(mario['screen_box'][0]+8),
                                 'dx_end':right_start-(mario['screen_box'][0]+8),
                                 'surface_dy':floor-sum(mario['screen_box'][1::2]),
                                 'kind':'visually_confirmed_floor_opening',
                                 'evidence':'separated observed ground support intervals'}
                    if not any(abs(g['dx_start']-candidate['dx_start']) < 2 and
                               abs(g['dx_end']-candidate['dx_end']) < 2 for g in gaps):
                        gaps.append(candidate)
        mx,my,mw,mh = mario['screen_box']
        def relative(obj):
            x,y,w,h = obj['box']
            result = {'kind':obj['kind'],'dx':round(x-mx-mw/2,1),'dy':round(y-my-mh,1),'width':w,'height':h}
            for key in ('id','vx','vy','source','last_seen_frame','age_frames','visibility','appearance','visible_fraction'):
                if key in obj:
                    result[key] = obj[key]
            return result
        merged = []
        for obj in sorted(terrain,key=lambda o:(o['kind'],o['source'],o['last_seen_frame'],o['visibility'],o['box'][1],o['box'][0])):
            if merged:
                prev=merged[-1]
                a,b=prev['box'],obj['box']
                if all(prev[k]==obj[k] for k in ('kind','source','last_seen_frame','visibility')) and a[1]==b[1] and a[3]==b[3] and a[0]+a[2]==b[0]:
                    a[2]+=b[2]
                    continue
            merged.append(deepcopy(obj))
        self.updates += 1
        observation = {'frame':frame_number,'mario':deepcopy(mario),
                       'camera':{'world_x_offset':camera,'delta_x':None if self.last_camera is None else camera-self.last_camera,
                                 'source':'Mario world_x minus screen_x'},
                       'terrain':[relative(o) for o in merged], 'sprites':[relative(o) for o in sprites],
                       'floor_gaps':gaps,'terrain_changes':changes,
                       'visible_dx':[-mx-mw/2,256-mx-mw/2],'outside_view':'unknown',
                       'perception':'Native SMB1 outdoor RGB appearance matching. Observed versus remembered are explicit; visually empty does not rule out invisible blocks. Unknown is not a pit.',
                       'observer_updates':self.updates,**grids}
        self.last_camera,self.last_frame = camera,frame_number
        self.last_detection_frame = frame_number
        self.cached_observation = deepcopy(observation)
        return observation,deepcopy(merged+sprites)
