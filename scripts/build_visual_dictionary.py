"""Build appearance templates from the installed game's static CHR artwork.

This reads graphic tiles, never a level map, sprite RAM or future game states.
Run locally after installing requirements. Art is not licensed for redistribution.
"""
from pathlib import Path
import gym_super_mario_bros
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

def main():
    rom = (Path(gym_super_mario_bros.__file__).parent / '_roms/super-mario-bros.nes').read_bytes()
    start = 16 + rom[4] * 16384
    raw = np.frombuffer(rom[start:start + rom[5]*8192], dtype=np.uint8).reshape(-1, 16)
    tiles = np.stack([np.unpackbits(t[:8]).reshape(8,8) +
                      2*np.unpackbits(t[8:]).reshape(8,8) for t in raw])
    palettes = {
        'terrain': [[104,136,252],[240,208,176],[228,92,16],[0,0,0]],
        'block': [[104,136,252],[252,160,68],[228,92,16],[0,0,0]],
        'enemy': [[104,136,252],[0,0,0],[240,208,176],[228,92,16]],
        'pipe': [[104,136,252],[184,248,24],[0,168,0],[0,0,0]],
    }
    # Each row lists CHR tile indices from left to right, top to bottom.
    definitions = {
        'goomba_walk': ('enemy', [[0x70,0x71],[0x72,0x73]]),
        'used': ('block', [[0x157,0x158],[0x159,0x15a]]),
        'stair': ('terrain', [[0x1ab,0x1ad],[0x1ac,0x1ae]]),
        'pipe_top': ('pipe', [[0x160,0x161,0x162,0x163], [0x164,0x165,0x166,0x167]]),
        'pipe_body': ('pipe', [[0x168,0x169,0x126,0x16a],[0x168,0x169,0x126,0x16a]]),
    }
    out = ROOT/'assets/templates'
    previews = []
    for name,(palette,indices) in definitions.items():
        indexed = np.block([[tiles[i] for i in row] for row in indices])
        rgba = np.concatenate([np.array(palettes[palette],dtype=np.uint8)[indexed],
                               ((indexed != 0)*255).astype(np.uint8)[...,None]],axis=2)
        im = Image.fromarray(rgba)
        im.save(out/f'{name}.png')
        previews.append((name,im))
        if name == 'goomba_walk':
            flipped = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped.save(out/'goomba_walk_mirror.png')
            previews.append(('goomba_walk_mirror',flipped))
    preview = Image.new('RGB',(512, len(previews)*100),'#6888fc')
    draw = ImageDraw.Draw(preview)
    for i,(name,im) in enumerate(previews):
        draw.text((4,i*100+4),name,fill='white')
        preview.paste(im.resize((im.width*4,im.height*4)),(260,i*100),im.resize((im.width*4,im.height*4)))
    dest = ROOT/'runs/atlas'
    dest.mkdir(parents=True,exist_ok=True)
    preview.save(dest/'dictionary.png')
    terrain = np.array(palettes['terrain'],dtype=np.uint8)[tiles]
    for name in ('ground','brick','question'):
        sample = np.array(Image.open(out/f'{name}.png').convert('RGB'))
        ids = []
        for y,x in ((0,0),(0,8),(8,0),(8,8)):
            errors = ((terrain.astype(float)-sample[y:y+8,x:x+8])**2).sum((1,2,3))
            best = int(errors.argmin())
            ids.append((hex(best),float(errors[best])))
        print(name,ids)

if __name__ == '__main__':
    main()
