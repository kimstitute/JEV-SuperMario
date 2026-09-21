from pathlib import Path
import json
from collections import Counter
import gymnasium as gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from PIL import Image

out = Path('runs/probe')
out.mkdir(parents=True, exist_ok=True)
env = JoypadSpace(gym.make('SuperMarioBros-1-1-v0'), SIMPLE_MOVEMENT)
frame, info = env.reset(seed=0)
for t in range(161):
    if t in (0, 40, 80, 120, 160):
        Image.fromarray(frame).save(out / f'{t:04}.png')
        core = env.unwrapped
        print(json.dumps({'t':t, 'info':info, 'self_ram': {hex(a):int(core.ram[a]) for a in [0x86,0x3ad,0x3b8,0x1d,0x57,0x9f,0x6d,0x71c,0x3b0,0xce]}}))
        if t == 0:
            print(Counter(map(tuple, frame.reshape(-1,3).tolist())).most_common(20))
    if t < 160:
        frame, _, terminated, truncated, info = env.step(1)
        if terminated or truncated:
            print('terminated', t)
            break
env.close()
