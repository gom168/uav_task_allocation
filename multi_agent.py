import numpy as np
from collections import Counter
import pygame
import os
from typing import List, Dict, Tuple, Optional

# --- Constants ---
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
FPS = 60
COMBAT_SIM_MINUTES = 20
FRAMES_PER_COMBAT_MINUTE = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED_COLOR = (200, 0, 0)
BLUE_COLOR = (0, 0, 200)
GREEN_COLOR = (0, 200, 0)
GREY_COLOR = (100, 100, 100)
DESTROYED_COLOR = (50, 50, 50)
BASE_RED = (200, 50, 50, 150)  # Semi-transparent
BASE_BLUE = (50, 50, 200, 150)

# Battlefield colors
BATTLEFIELD_COLORS = [
    (0, 150, 0),  # Green
    (150, 0, 150),  # Purple
    (0, 150, 150),  # Cyan
    (150, 150, 0),  # Yellow
    (150, 0, 0)  # Dark red
]

# Fonts
FONT_SIZE_SMALL = 20
FONT_SIZE_NORMAL = 24
FONT_SIZE_LARGE = 36

# UAV parameters
UAV_SIZE = 30
UAV_SPEED = 4
UAV_ENGAGEMENT_RADIUS = 150
BATTLEFIELD_MARKER_RADIUS = 20

# Combat effects
LOG_DISPLAY_TIME_MS = 1000
MAX_LOG_MESSAGES = 5
ATTACK_EFFECT_DISPLAY_TIME_MS = 200


class UAV:
    """Represents a single UAV"""

    def __init__(self, uav_type: str, team: str, pos: pygame.math.Vector2,
                 image: pygame.Surface, battlefield_id: int, is_render_mode: bool):
        self.type = uav_type
        self.team = team  # 'red' or 'blue'
        self.pos = pos
        self.battlefield_id = battlefield_id
        self.is_alive = True
        self.cooldowns = {'mw_laser': 0, 'net': 0}
        if image is None and is_render_mode:
            # 创建默认图像
            color = RED_COLOR if team == 'red' else BLUE_COLOR
            self.image = self._create_default_image(color)
        else:
            self.image = image
        self.is_render_mode = is_render_mode

        if is_render_mode:
            self.rect = image.get_rect(topleft=self.pos) if image else None
        else:
            # Simplified rect for non-render mode
            class SimpleRect:
                def __init__(self, x, y, width, height):
                    self.x = x
                    self.y = y
                    self.width = width
                    self.height = height
                    self.topleft = (x, y)

                @property
                def x(self): return self._x

                @x.setter
                def x(self, value): self._x = value

                @property
                def y(self): return self._y

                @y.setter
                def y(self, value): self._y = value

                @property
                def topleft(self): return (self._x, self._y)

                @topleft.setter
                def topleft(self, value):
                    self._x, self._y = value

            self.rect = SimpleRect(self.pos.x, self.pos.y, UAV_SIZE, UAV_SIZE)

        self.target_pos = None
        self.arrival_threshold = UAV_SPEED * 2

    def _create_default_image(self, color: Tuple[int, int, int]) -> pygame.Surface:
        """Create a default UAV image"""
        surface = pygame.Surface((UAV_SIZE, UAV_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (UAV_SIZE // 2, UAV_SIZE // 2), UAV_SIZE // 2)
        return surface

    def set_target_pos(self, target_pos: pygame.math.Vector2):
        self.target_pos = target_pos

    def move_towards(self, speed: float):
        if not self.is_alive or self.target_pos is None:
            return

        direction = self.target_pos - self.pos
        distance = direction.length()

        if distance > self.arrival_threshold:
            self.pos += direction.normalize() * min(speed, distance)
        else:
            self.pos = self.target_pos
            self.target_pos = None

        self.rect.topleft = self.pos

    def update_cooldowns(self, delta_time: int = 1):
        for weapon_type in self.cooldowns:
            if self.cooldowns[weapon_type] > 0:
                self.cooldowns[weapon_type] -= delta_time

    def can_fire(self, weapon_type: str) -> bool:
        return self.is_alive and self.cooldowns.get(weapon_type, 0) == 0


class MultiBattlefieldEnv:
    """Multi-battlefield UAV combat environment"""

    def __init__(self,
                 num_battlefields: int = 3,
                 render_mode: bool = True,
                 background_image: Optional[str] = None,
                 red_images: Dict[str, str] = None,
                 blue_images: Dict[str, str] = None):

        self.num_battlefields = num_battlefields
        self.render_mode = render_mode

        # Initial configuration
        self.initial_blue_uavs = Counter({
            'ground_attack': 10,
            'recon': 10,
            'escort': 30
        })
        self.initial_red_uavs = Counter({
            'interceptor': 40,
            'recon': 10,
            'escort': 0
        })

        # Combat parameters
        self.PROB_RED_MW_LASER_DMG_BLUE_GA = 0.7
        self.PROB_RED_INTERCEPTOR_NET_DMG_BLUE_GA = 0.9
        self.PROB_BLUE_ESCORT_DMG_RED_INTERCEPTOR_MW_LASER = 0.7

        self.INTERVAL_MW_LASER_RED = 1
        self.INTERVAL_MW_LASER_BLUE = 3
        self.INTERVAL_NET_RED = 5

        # Base rectangles
        self.red_base_rect = pygame.Rect(50, SCREEN_HEIGHT // 2 - 150, 200, 300)
        self.blue_base_rect = pygame.Rect(SCREEN_WIDTH - 250, SCREEN_HEIGHT // 2 - 150, 200, 300)

        # Initialize Pygame
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption(f"Multi-Battlefield UAV Combat ({num_battlefields} battlefields)")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)
            self.font_normal = pygame.font.Font(None, FONT_SIZE_NORMAL)
            self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)

            # Load images
            self.images = {
                'background': self._load_background(background_image),
                'red': {
                    'interceptor': self._load_uav_image(red_images.get('interceptor') if red_images else None,
                                                        RED_COLOR),
                    'recon': self._load_uav_image(red_images.get('recon') if red_images else None, RED_COLOR),
                    'escort': self._load_uav_image(red_images.get('escort') if red_images else None, RED_COLOR)
                },
                'blue': {
                    'ground_attack': self._load_uav_image(blue_images.get('ground_attack') if blue_images else None,
                                                          BLUE_COLOR),
                    'recon': self._load_uav_image(blue_images.get('recon') if blue_images else None, BLUE_COLOR),
                    'escort': self._load_uav_image(blue_images.get('escort') if blue_images else None, BLUE_COLOR)
                }
            }
        else:
            self.screen = None
            self.clock = None
            self.images = {
                'background': None,
                'red': {
                    'interceptor': self._load_uav_image(None, RED_COLOR),
                    'recon': self._load_uav_image(None, RED_COLOR),
                    'escort': self._load_uav_image(None, RED_COLOR)
                },
                'blue': {
                    'ground_attack': self._load_uav_image(None, BLUE_COLOR),
                    'recon': self._load_uav_image(None, BLUE_COLOR),
                    'escort': self._load_uav_image(None, BLUE_COLOR)
                }
            }

        # Calculate battlefield positions
        # 注释
        # self.battlefield_positions = self._calculate_battlefield_positions()

        # Reset environment
        self.reset()

    def _load_background(self, path: Optional[str]) -> Optional[pygame.Surface]:
        """Load background image with error handling"""
        if not path or not os.path.exists(path):
            return None

        try:
            img = pygame.image.load(path).convert()
            return pygame.transform.scale(img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        except pygame.error as e:
            print(f"Warning: Could not load background image: {e}")
            return None

    def _load_uav_image(self, path: Optional[str], default_color: Tuple[int, int, int]) -> Optional[pygame.Surface]:
        """Load UAV image with fallback to default"""
        if not path or not os.path.exists(path):
            return self._create_default_image(default_color) if self.render_mode else None

        try:
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (UAV_SIZE, UAV_SIZE))
        except pygame.error as e:
            print(f"Warning: Could not load UAV image from {path}: {e}")
            return self._create_default_image(default_color) if self.render_mode else None

    def _create_default_image(self, color: Tuple[int, int, int]) -> pygame.Surface:
        """Create a default UAV image"""
        surface = pygame.Surface((UAV_SIZE, UAV_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (UAV_SIZE // 2, UAV_SIZE // 2), UAV_SIZE // 2)
        return surface

    def _create_default_image(self, color: Tuple[int, int, int]) -> pygame.Surface:
        """Create a default UAV image"""
        if not self.render_mode:
            return None

        surface = pygame.Surface((UAV_SIZE, UAV_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (UAV_SIZE // 2, UAV_SIZE // 2), UAV_SIZE // 2)
        return surface

    # 注释
    # def _calculate_battlefield_positions(self) -> List[Tuple[int, int]]:
    #     """Calculate positions for all battlefields"""
    #     positions = []
    #     margin = 100
    #     spacing_x = (SCREEN_WIDTH - 2 * margin) // (self.num_battlefields + 1)

    #     for i in range(self.num_battlefields):
    #         x = margin + (i + 1) * spacing_x
    #         y = SCREEN_HEIGHT // 2
    #         positions.append((x, y))

    #     return positions

    def _get_battlefield_positions(self, battlefield_coords: List[tuple]):
        """Calculate positions for all battlefields"""
        positions = []
        for i in range(self.num_battlefields):
            x = battlefield_coords[i][0]
            y = battlefield_coords[i][1]
            positions.append((x, y))

        self.battlefield_positions = positions

    def reset(self):
        """Reset the environment"""
        self.red_uavs = self.initial_red_uavs.copy()
        self.blue_uavs = self.initial_blue_uavs.copy()
        self.time_step = 0
        self.global_done = False

        # UAVs engaged in each battlefield
        self.red_engaged_uavs = [[] for _ in range(self.num_battlefields)]
        self.blue_engaged_uavs = [[] for _ in range(self.num_battlefields)]

        # Combat effects
        self.explosion_effects = []
        self.active_attack_lines = []
        self.combat_log = []

        # Initial state
        initial_state = []
        for i in range(self.num_battlefields):
            initial_state.append({
                'battlefield_id': i,
                'friendly_remaining': dict(self.red_uavs),
                'enemy_remaining': dict(self.blue_uavs),
                'active': False
            })

        if self.render_mode:
            self.render(initial_state)

        return initial_state

    def _get_blue_deployment(self, battlefield_id: int) -> Dict[str, int]:
        """更智能的蓝方部署策略"""
        # 基础约束：至少1侦察机+1攻击机
        min_recon = 1
        min_attack = 1

        # 动态调整最大出动量（避免前期倾巢而出）
        max_recon = max(min_recon, min(self.blue_uavs['recon'] // 3 + 1, self.blue_uavs['recon']))
        max_attack = max(min_attack, min(self.blue_uavs['ground_attack'] // 3 + 1, self.blue_uavs['ground_attack']))
        max_escort = min(self.blue_uavs['escort'] // 3 + 1, self.blue_uavs['escort'])

        recon = np.random.randint(1, max(2, max_recon))
        attack = np.random.randint(1, max(2, max_attack))
        escort = np.random.randint(0, max(1, max_escort))

        if max_recon <= 2:
            recon = max_recon

        if max_attack <= 2:
            attack = max_attack

        if escort <= 2:
            escort = max_escort

        # 确保不超过剩余兵力
        return {
            'recon': min(recon, self.blue_uavs['recon']),
            'ground_attack': min(attack, self.blue_uavs['ground_attack']),
            'escort': min(escort, self.blue_uavs['escort'])
        }

    def step(self, actions: List[Dict], battlefield_coords: List[tuple]):
        # def step(self, action: dict, enemy_formation: dict, battlefield_coords: tuple):
        """
        battlefield_positions
        Execute one step with actions for each battlefield
        Returns: next_state, rewards, done, info
        """
        if self.global_done:
            return self._get_state(), [0] * self.num_battlefields, True, {}
        # 增加对传入战场位置的输入
        self._get_battlefield_positions(battlefield_coords)

        # 1. Red deployment
        for i, action in enumerate(actions):
            # if not self._is_battlefield_active(i):  # Only deploy to inactive battlefields
            # Ensure constraints: at least 1 recon and 1 interceptor
            recon = min(action.get('recon', 0), self.red_uavs['recon'])
            interceptor = min(action.get('interceptor', 0), self.red_uavs['interceptor'])
            escort = min(action.get('escort', 0), self.red_uavs['escort'])

            if recon < 1 and self.red_uavs['recon'] > 0:
                recon = 1
            if interceptor < 1 and self.red_uavs['interceptor'] > 0:
                interceptor = 1

            self._deploy_red_uavs(i, interceptor, recon, escort)
            self.red_uavs['recon'] -= recon
            self.red_uavs['interceptor'] -= interceptor
            self.red_uavs['escort'] -= escort

        # 2. Blue deployment (simple AI)
        for i in range(self.num_battlefields):
            # if not self._is_battlefield_active(i) and len(self.red_engaged_uavs[i]) > 0:
            deployment = self._get_blue_deployment(i)
            self._deploy_blue_uavs(i, deployment['ground_attack'], deployment['recon'], deployment['escort'])
            for k, v in deployment.items():
                self.blue_uavs[k] -= v

        # 3. Simulate combat in all battlefields
        rewards = [0] * self.num_battlefields
        total_frames = COMBAT_SIM_MINUTES * FRAMES_PER_COMBAT_MINUTE

        for frame in range(total_frames):
            current_minute = frame // FRAMES_PER_COMBAT_MINUTE + 1

            # Update UAV positions
            for i in range(self.num_battlefields):
                for uav in self.red_engaged_uavs[i] + self.blue_engaged_uavs[i]:
                    uav.move_towards(UAV_SPEED)

            # Check weapon cooldowns every minute
            if frame % FRAMES_PER_COMBAT_MINUTE == 0:
                for i in range(self.num_battlefields):
                    if self._is_battlefield_active(i):
                        reward = self._simulate_combat(i, current_minute)
                        rewards[i] += reward

            # Check if all battles have ended
            if all(not self._is_battlefield_active(i) for i in range(self.num_battlefields)):
                break

            # Render
            if self.render_mode:
                self.render(self._get_state())
                self.clock.tick(FPS)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return self._get_state(), rewards, True, {}

        # Check global done condition
        self.global_done = (self.blue_uavs['ground_attack'] <= 0 and self.blue_uavs['escort'] < 0 and self.blue_uavs[
            'recon'] <= 0) or (sum(self.blue_uavs.values()) <= 0)

        return self._get_state(), rewards, self.global_done, {}

    def _is_battlefield_active(self, battlefield_id: int) -> bool:
        """Check if a battlefield has active combat"""
        return (len(self.red_engaged_uavs[battlefield_id]) > 0 and
                len(self.blue_engaged_uavs[battlefield_id]) > 0 and
                any(u.is_alive for u in self.red_engaged_uavs[battlefield_id]) and
                any(u.is_alive for u in self.blue_engaged_uavs[battlefield_id]))

    def _deploy_red_uavs(self, battlefield_id: int, interceptors: int, recons: int, escorts: int):
        """Deploy red UAVs to a battlefield"""
        battlefield_pos = self.battlefield_positions[battlefield_id]

        for _ in range(interceptors):
            pos = pygame.math.Vector2(
                np.random.uniform(self.red_base_rect.left, self.red_base_rect.right - UAV_SIZE),
                np.random.uniform(self.red_base_rect.top, self.red_base_rect.bottom - UAV_SIZE)
            )
            target_pos = pygame.math.Vector2(battlefield_pos[0] - 50, battlefield_pos[1])
            if self.render_mode == True:
                uav = UAV('interceptor', 'red', pos,
                          self.images['red']['interceptor'], battlefield_id, self.render_mode)
            else:
                uav = UAV('interceptor', 'red', pos,
                          None, battlefield_id, self.render_mode)
            uav.set_target_pos(target_pos)
            self.red_engaged_uavs[battlefield_id].append(uav)

        for _ in range(recons):
            pos = pygame.math.Vector2(
                np.random.uniform(self.red_base_rect.left, self.red_base_rect.right - UAV_SIZE),
                np.random.uniform(self.red_base_rect.top, self.red_base_rect.bottom - UAV_SIZE)
            )
            target_pos = pygame.math.Vector2(battlefield_pos[0] - 50, battlefield_pos[1] + 30)
            uav = UAV('recon', 'red', pos,
                      self.images['red']['recon'], battlefield_id, self.render_mode)
            uav.set_target_pos(target_pos)
            self.red_engaged_uavs[battlefield_id].append(uav)

        for _ in range(escorts):
            pos = pygame.math.Vector2(
                np.random.uniform(self.red_base_rect.left, self.red_base_rect.right - UAV_SIZE),
                np.random.uniform(self.red_base_rect.top, self.red_base_rect.bottom - UAV_SIZE)
            )
            target_pos = pygame.math.Vector2(battlefield_pos[0] - 50, battlefield_pos[1] - 30)
            uav = UAV('escort', 'red', pos,
                      self.images['red']['escort'], battlefield_id, self.render_mode)
            uav.set_target_pos(target_pos)
            self.red_engaged_uavs[battlefield_id].append(uav)

    def _deploy_blue_uavs(self, battlefield_id: int, attacks: int, recons: int, escorts: int):
        """Deploy blue UAVs to a battlefield"""
        battlefield_pos = self.battlefield_positions[battlefield_id]

        for _ in range(attacks):
            pos = pygame.math.Vector2(
                np.random.uniform(self.blue_base_rect.left, self.blue_base_rect.right - UAV_SIZE),
                np.random.uniform(self.blue_base_rect.top, self.blue_base_rect.bottom - UAV_SIZE)
            )
            target_pos = pygame.math.Vector2(battlefield_pos[0] + 50, battlefield_pos[1])
            uav = UAV('ground_attack', 'blue', pos,
                      self.images['blue']['ground_attack'], battlefield_id, self.render_mode)
            uav.set_target_pos(target_pos)
            self.blue_engaged_uavs[battlefield_id].append(uav)

        for _ in range(recons):
            pos = pygame.math.Vector2(
                np.random.uniform(self.blue_base_rect.left, self.blue_base_rect.right - UAV_SIZE),
                np.random.uniform(self.blue_base_rect.top, self.blue_base_rect.bottom - UAV_SIZE)
            )
            target_pos = pygame.math.Vector2(battlefield_pos[0] + 50, battlefield_pos[1] + 30)
            uav = UAV('recon', 'blue', pos,
                      self.images['blue']['recon'], battlefield_id, self.render_mode)
            uav.set_target_pos(target_pos)
            self.blue_engaged_uavs[battlefield_id].append(uav)

        for _ in range(escorts):
            pos = pygame.math.Vector2(
                np.random.uniform(self.blue_base_rect.left, self.blue_base_rect.right - UAV_SIZE),
                np.random.uniform(self.blue_base_rect.top, self.blue_base_rect.bottom - UAV_SIZE)
            )
            target_pos = pygame.math.Vector2(battlefield_pos[0] + 50, battlefield_pos[1] - 30)
            uav = UAV('escort', 'blue', pos,
                      self.images['blue']['escort'], battlefield_id, self.render_mode)
            uav.set_target_pos(target_pos)
            self.blue_engaged_uavs[battlefield_id].append(uav)

    def _simulate_combat(self, battlefield_id: int, current_minute: int) -> float:
        """Simulate combat in one battlefield, return reward"""
        reward = 0.0
        battlefield_pos = self.battlefield_positions[battlefield_id]
        battlefield_center = pygame.math.Vector2(battlefield_pos)

        # Get UAVs within engagement radius
        active_red = [u for u in self.red_engaged_uavs[battlefield_id]
                      if u.is_alive and u.pos.distance_to(battlefield_center) <= UAV_ENGAGEMENT_RADIUS]
        active_blue = [u for u in self.blue_engaged_uavs[battlefield_id]
                       if u.is_alive and u.pos.distance_to(battlefield_center) <= UAV_ENGAGEMENT_RADIUS]

        # Red MW/Laser attack (every minute)
        if current_minute % self.INTERVAL_MW_LASER_RED == 0:
            firing_red = [u for u in active_red if u.type in ['interceptor', 'escort'] and u.can_fire('mw_laser')]
            if firing_red and active_blue:
                hits = self._calculate_damage(len(firing_red), self.PROB_RED_MW_LASER_DMG_BLUE_GA)
                blue_losses = min(hits, len(active_blue))

                for i in range(blue_losses):
                    if active_blue[i].is_alive:
                        active_blue[i].is_alive = False
                        self.explosion_effects.append((active_blue[i].pos.copy(), pygame.time.get_ticks()))

                        # Reward based on destroyed UAV type
                        if active_blue[i].type == 'ground_attack':
                            reward += 3.0
                        elif active_blue[i].type == 'recon':
                            reward += 2.0
                        else:
                            reward += 1.0

                        # Add attack effect
                        if self.render_mode:
                            attacker = np.random.choice(firing_red)
                            self.active_attack_lines.append((
                                attacker.pos + pygame.math.Vector2(UAV_SIZE / 2, UAV_SIZE / 2),
                                active_blue[i].pos + pygame.math.Vector2(UAV_SIZE / 2, UAV_SIZE / 2),
                                (255, 255, 0),  # Yellow laser
                                pygame.time.get_ticks() + ATTACK_EFFECT_DISPLAY_TIME_MS
                            ))

                # Set cooldown
                for uav in firing_red:
                    uav.cooldowns['mw_laser'] = self.INTERVAL_MW_LASER_RED

                # Log
                if blue_losses > 0 and self.render_mode:
                    self.combat_log.append((
                        f"Battlefield {battlefield_id}: Red destroyed {blue_losses} Blue UAVs",
                        pygame.time.get_ticks() + LOG_DISPLAY_TIME_MS
                    ))

        # Blue MW/Laser attack (every 3 minutes)
        if current_minute % self.INTERVAL_MW_LASER_BLUE == 0:
            firing_blue = [u for u in active_blue if u.type == 'escort' and u.can_fire('mw_laser')]
            if firing_blue and active_red:
                hits = self._calculate_damage(len(firing_blue), self.PROB_BLUE_ESCORT_DMG_RED_INTERCEPTOR_MW_LASER)
                red_losses = min(hits, len(active_red))

                for i in range(red_losses):
                    if active_red[i].is_alive:
                        active_red[i].is_alive = False
                        self.explosion_effects.append((active_red[i].pos.copy(), pygame.time.get_ticks()))
                        reward -= 0.5  # Penalty for red losses

                        # Add attack effect
                        if self.render_mode:
                            attacker = np.random.choice(firing_blue)
                            self.active_attack_lines.append((
                                attacker.pos + pygame.math.Vector2(UAV_SIZE / 2, UAV_SIZE / 2),
                                active_red[i].pos + pygame.math.Vector2(UAV_SIZE / 2, UAV_SIZE / 2),
                                (255, 255, 0),  # Yellow laser
                                pygame.time.get_ticks() + ATTACK_EFFECT_DISPLAY_TIME_MS
                            ))

                # Set cooldown
                for uav in firing_blue:
                    uav.cooldowns['mw_laser'] = self.INTERVAL_MW_LASER_BLUE

                # Log
                if red_losses > 0 and self.render_mode:
                    self.combat_log.append((
                        f"Battlefield {battlefield_id}: Blue destroyed {red_losses} Red UAVs",
                        pygame.time.get_ticks() + LOG_DISPLAY_TIME_MS
                    ))

        return reward

    def _calculate_damage(self, attacker_count: int, damage_prob: float) -> int:
        """Calculate number of successful hits"""
        return sum(1 for _ in range(attacker_count) if np.random.rand() < damage_prob)

    def _get_state(self) -> List[Dict]:
        """Get current state of all battlefields"""
        states = []
        for i in range(self.num_battlefields):
            states.append({
                'battlefield_id': i,
                'friendly_remaining': dict(self.red_uavs),
                'enemy_remaining': dict(self.blue_uavs),
                # 'active': self._is_battlefield_active(i)
            })
        return states

    def render(self, states: List[Dict]):
        """Render the environment"""
        if not self.render_mode:
            return

        # Clear screen
        if self.images['background']:
            self.screen.blit(self.images['background'], (0, 0))
        else:
            self.screen.fill(GREY_COLOR)

        # Draw bases
        self._draw_base(self.red_base_rect, "Red Base", BASE_RED)
        self._draw_base(self.blue_base_rect, "Blue Base", BASE_BLUE)

        # Draw battlefields
        for i, pos in enumerate(self.battlefield_positions):
            color = BATTLEFIELD_COLORS[i % len(BATTLEFIELD_COLORS)]
            pygame.draw.circle(self.screen, color, pos, BATTLEFIELD_MARKER_RADIUS, 2)
            text = self.font_small.render(f"Battlefield {i}", True, color)
            self.screen.blit(text, (pos[0] - text.get_width() // 2, pos[1] - BATTLEFIELD_MARKER_RADIUS - 25))
            pygame.draw.circle(self.screen, (*color, 50), pos, UAV_ENGAGEMENT_RADIUS, 1)

        # Draw UAVs
        for i in range(self.num_battlefields):
            for uav in self.red_engaged_uavs[i] + self.blue_engaged_uavs[i]:
                if uav.is_alive:
                    self.screen.blit(uav.image, uav.pos)
                else:
                    # Draw destroyed UAV
                    destroyed_img = pygame.Surface((UAV_SIZE, UAV_SIZE), pygame.SRCALPHA)
                    destroyed_img.fill((*DESTROYED_COLOR, 150))
                    self.screen.blit(destroyed_img, uav.pos)
                    pygame.draw.line(self.screen, BLACK, uav.pos,
                                     (uav.pos.x + UAV_SIZE, uav.pos.y + UAV_SIZE), 2)
                    pygame.draw.line(self.screen, BLACK,
                                     (uav.pos.x + UAV_SIZE, uav.pos.y),
                                     (uav.pos.x, uav.pos.y + UAV_SIZE), 2)

        # Draw combat effects
        current_time = pygame.time.get_ticks()

        # Explosions
        active_explosions = []
        for pos, start_time in self.explosion_effects:
            if current_time - start_time < 500:
                radius = int((current_time - start_time) / 500 * (UAV_SIZE * 1.5))
                pygame.draw.circle(self.screen, (255, 165, 0),
                                   pos + pygame.math.Vector2(UAV_SIZE / 2, UAV_SIZE / 2),
                                   radius, 2)
                active_explosions.append((pos, start_time))
        self.explosion_effects = active_explosions

        # Attack lines
        active_lines = []
        for start_pos, end_pos, color, end_time in self.active_attack_lines:
            if current_time < end_time:
                pygame.draw.line(self.screen, color, start_pos, end_pos, 3)
                active_lines.append((start_pos, end_pos, color, end_time))
        self.active_attack_lines = active_lines

        # Draw remaining forces
        text_red = self.font_normal.render(
            f"Red: Interceptors={self.red_uavs['interceptor']} Recon={self.red_uavs['recon']}",
            True, RED_COLOR)
        text_blue = self.font_normal.render(
            f"Blue: Attack={self.blue_uavs['ground_attack']} Escort={self.blue_uavs['escort']}",
            True, BLUE_COLOR)
        self.screen.blit(text_red, (20, 20))
        self.screen.blit(text_blue, (SCREEN_WIDTH - text_blue.get_width() - 20, 20))

        # Draw combat log
        y_log = SCREEN_HEIGHT - 120
        display_log = sorted(self.combat_log, key=lambda x: x[1])[-MAX_LOG_MESSAGES:]
        for i, (msg, display_until) in enumerate(display_log):
            if current_time < display_until:
                log_text = self.font_small.render(msg, True, WHITE)
                self.screen.blit(log_text, (20, y_log + i * (FONT_SIZE_SMALL + 5)))

        # Update display
        pygame.display.flip()

    def _draw_base(self, rect: pygame.Rect, name: str, color: Tuple[int, int, int, int]):
        """Draw a base rectangle with label"""
        base_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        base_surface.fill(color)
        self.screen.blit(base_surface, rect.topleft)
        pygame.draw.rect(self.screen, color[:3], rect, 2)  # Outline

        # Draw base name
        text = self.font_normal.render(name, True, WHITE)
        self.screen.blit(text, (rect.x + 10, rect.y + 10))

    def close(self):
        """Close the environment"""
        if self.render_mode:
            pygame.quit()


if __name__ == "__main__":

    image_config = {
        'background_image': 'figure/背景.png',
        'red_images': {
            'interceptor': 'figure/红.png',
            'recon': 'figure/红2.png',
            'escort': 'figure/红3.png'
        },
        'blue_images': {
            'ground_attack': 'figure/蓝.png',
            'recon': 'figure/蓝2.png',
            'escort': 'figure/蓝3.png'
        }
    }

    # Create environment
    env = MultiBattlefieldEnv(
        num_battlefields=3,
        render_mode=True,
        **image_config
    )


    # Simple policy for each battlefield
    def policy(state: List[Dict]) -> List[Dict]:
        actions = []
        for s in state:
            if not s['active']:
                # Deploy 3 interceptors and 1 recon if available
                interceptors = min(3, s['friendly_remaining']['interceptor'])
                recons = min(1, s['friendly_remaining']['recon'])
                actions.append({'interceptor': interceptors, 'recon': recons, 'escort': 0})
            else:
                actions.append({})  # No deployment to active battlefields
        return actions


    # Run simulation
    state = env.reset()
    done = False

    while not done:
        action = policy(state)
        next_state, rewards, done, _ = env.step(action)
        print(f"Step rewards: {rewards}")
        state = next_state

        # Add small delay for visualization
        if env.render_mode:
            pygame.time.wait(500)

    env.close()