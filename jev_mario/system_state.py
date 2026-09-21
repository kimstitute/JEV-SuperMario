"""Canonical emulator/RAM state for the realtime Jev controller.

RGB perception remains available for debug and regression comparison, but this
object is the model-facing source of player, enemy and local collision facts.
The addresses follow the NES layout used by the public TypeSafe Mario parser.
"""
from copy import deepcopy

ENEMY_NAMES = {0x06: 'goomba', 0x05: 'hammer_bro', 0x07: 'bloober',
               0x0E: 'cheep_cheep', 0x12: 'piranha_plant', 0x2D: 'bowser',
               0x31: 'flagpole'}


class SystemStateParser:
    def __init__(self):
        self.last_x = None
        self.last_y = None
        self.last_enemy_dx = {}
        self.best_x = 0
        self.stalled = 0
        self.airborne_frames = 0
        self.last_grounded = None

    @staticmethod
    def byte(ram, address, default=0):
        return int(ram[address]) if ram is not None and 0 <= address < len(ram) else default

    def parse(self, info, ram, mario, frame, action, reward=0.0, response_delay_frames=0):
        x = int(info.get('x_pos', mario.get('world_x', 0)))
        y = int(info.get('y_pixel', mario.get('feet_y', 0)))
        dx = 0 if self.last_x is None else x - self.last_x
        dy = 0 if self.last_y is None else y - self.last_y
        grounded = bool(mario.get('grounded'))
        self.best_x = max(self.best_x, x)
        self.stalled = self.stalled + 1 if self.last_x is not None and x <= self.last_x else 0
        self.airborne_frames = 0 if grounded else self.airborne_frames + 1
        enemies = []
        for slot in range(5):
            active = self.byte(ram, 0x000F + slot)
            kind_id = self.byte(ram, 0x0016 + slot)
            if active == 0 or kind_id == 0:
                continue
            enemy_x = self.byte(ram, 0x006E + slot) * 256 + self.byte(ram, 0x0087 + slot)
            enemy_y = self.byte(ram, 0x00CF + slot)
            relative_x = enemy_x - x
            if not -192 <= relative_x <= 320:
                continue
            previous = self.last_enemy_dx.get(slot)
            relative_vx = 0 if previous is None else relative_x - previous
            self.last_enemy_dx[slot] = relative_x
            enemies.append({'slot': slot, 'kind_id': kind_id,
                            'kind': ENEMY_NAMES.get(kind_id, f'enemy_0x{kind_id:02x}'),
                            'relative_x_pixels': relative_x,
                            'relative_y_pixels': enemy_y - int(mario.get('feet_y', y)),
                            'relative_velocity_x': relative_vx,
                            'horizontal_relation': 'ahead' if relative_x >= 0 else 'behind'})
        grid = self._grid(ram, x, int(mario.get('feet_y', y)), enemies)
        supported = self._support(grid)
        terrain = self._terrain_features(grid)
        state = {
            'source': 'emulator_info_and_nes_ram',
            'player': {'x': x, 'y': y, 'dx': dx, 'dy': dy,
                       'grounded': grounded, 'jump_phase': 'grounded' if grounded else 'airborne',
                       'powerup_status': info.get('status', 'small')},
            'trajectory': {'airborne_frames': self.airborne_frames,
                           'crossing_known_gap': bool(terrain['gap_ahead']),
                           'gap_width_tiles': terrain['gap_width_tiles_visible']},
            'hazard': {'upcoming_enemies': enemies[:3], 'nearest_enemy': enemies[0] if enemies else None},
            'terrain': terrain,
            'reaction_timing': {'action_horizon_frames': 6,
                                'last_inference_delay_frames': response_delay_frames,
                                'total_reaction_horizon_frames': 6 + response_delay_frames},
            'recent_control': {'action': action, 'reward': float(reward),
                               'progress_gained_pixels': dx},
            'episode': {'progress': x, 'best_progress': self.best_x,
                        'stalled_frames': self.stalled,
                        'dead': bool(info.get('death', False)),
                        'stage_clear': bool(info.get('flag_get', False))},
            'local_grid': grid,
            'frame': frame,
            'ground_support_detected': supported,
        }
        self.last_x, self.last_y = x, y
        self.last_grounded = grounded
        return state

    @staticmethod
    def _grid(ram, x, y, enemies):
        if ram is None:
            return []
        cells = [['.' for _ in range(11)] for _ in range(9)]
        for row, dy in enumerate(range(-4, 5)):
            for col, dx in enumerate(range(-2, 9)):
                sx, sy = x + dx * 16, y + dy * 16
                page = (sx // 256) % 2
                sub_x, sub_y = (sx % 256) // 16, (sy - 32) // 16
                if 0 <= sub_y < 13 and SystemStateParser.byte(ram, 0x0500 + page * 208 + sub_y * 16 + sub_x):
                    cells[row][col] = '#'
        cells[4][2] = 'M'
        for enemy in enemies:
            col, row = 2 + round(enemy['relative_x_pixels'] / 16), 4 + round(enemy['relative_y_pixels'] / 16)
            if 0 <= row < 9 and 0 <= col < 11 and cells[row][col] == '.':
                cells[row][col] = 'E'
        return [''.join(row) for row in cells]

    @staticmethod
    def _support(grid):
        if not grid:
            return False
        return any(row[2] == '#' for row in grid[5:7])

    @staticmethod
    def _terrain_features(grid):
        if not grid:
            return {'geometry_available': False, 'gap_ahead': False,
                    'gap_width_tiles_visible': 0, 'obstacle_ahead': False}
        ground = next((r for r in range(5, len(grid)) if grid[r][2] == '#'), None)
        gap = None; width = 0; obstacle = None; height = 0
        if ground is not None:
            for col in range(3, len(grid[0])):
                supported = grid[ground][col] == '#'
                if not supported and gap is None:
                    gap = col - 2
                if gap is not None and not supported:
                    width += 1
                block_height = sum(grid[r][col] == '#' for r in range(max(0, ground - 4), ground))
                if block_height and obstacle is None:
                    obstacle, height = col - 2, block_height
        return {'geometry_available': True, 'gap_ahead': gap is not None and gap <= 3,
                'gap_distance_tiles': gap, 'gap_width_tiles_visible': width,
                'obstacle_ahead': obstacle is not None and obstacle <= 3,
                'obstacle_distance_tiles': obstacle, 'obstacle_height_tiles': height,
                'observation_reliability': 'ram_collision_grid'}
