import numpy as np
from collections import Counter
import pygame
import os  # For checking file existence

# --- Constants for Pygame Visualization ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60  # Higher FPS for smoother animation
COMBAT_SIM_MINUTES = 5  # Total combat duration in minutes
FRAMES_PER_COMBAT_MINUTE = 30  # How many frames to render per simulated minute

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED_COLOR = (200, 0, 0)
BLUE_COLOR = (0, 0, 200)
GREEN_COLOR = (0, 200, 0)
GREY_COLOR = (100, 100, 100)
DESTROYED_COLOR = (50, 50, 50)  # Dark grey for destroyed units
BATTLEFIELD_COLOR = (0, 150, 0)  # Green for battlefield area

FONT_SIZE_SMALL = 20
FONT_SIZE_NORMAL = 24``
FONT_SIZE_LARGE = 36

# Default image paths (can be customized)
DEFAULT_BACKGROUND_IMAGE = None
DEFAULT_RED_INTERCEPTOR_IMAGE = None
DEFAULT_RED_RECON_IMAGE = None
DEFAULT_RED_ESCORT_IMAGE = None

DEFAULT_BLUE_GROUND_ATTACK_IMAGE = None
DEFAULT_BLUE_RECON_IMAGE = None
DEFAULT_BLUE_ESCORT_IMAGE = None

# UAV representation sizes and movement
UAV_SIZE = 40  # Size for drawing UAVs (Reduced from 100 for better visibility of multiple units)
UAV_SPEED = 5  # Pixels per frame (adjust for faster/slower movement)

# Combat log settings
LOG_DISPLAY_TIME_MS = 1000  # How long log messages stay on screen (1000 ms = 1 second)
MAX_LOG_MESSAGES = 5

# --- Default Base and Initial UAV Configurations ---
# Default UAV counts if not provided during environment creation
DEFAULT_INITIAL_BLUE_UAVS = Counter({
    'ground_attack': 10,
    'recon': 10,
    'escort': 30
})
DEFAULT_INITIAL_RED_UAVS = Counter({
    'interceptor': 40,
    'recon': 10,
    'escort': 0
})

# Default Base Rectangles if not provided during environment creation
DEFAULT_RED_BASE_RECT = pygame.Rect(50, SCREEN_HEIGHT / 2 - 150, 200, 300)
DEFAULT_BLUE_BASE_RECT = pygame.Rect(SCREEN_WIDTH - 250, SCREEN_HEIGHT / 2 - 150, 200, 300)

BATTLEFIELD_MARKER_RADIUS = 30  # Radius for drawing battlefield center marker
UAV_ENGAGEMENT_RADIUS = 200  # Radius around battlefield center for UAVs to be considered "engaged"

# Cooldown for rendering attack lines/effects
ATTACK_EFFECT_DISPLAY_TIME_MS = 200  # How long to show a laser/net line (0.2 seconds)


class UAV:
    """Represents an individual Unmanned Aerial Vehicle."""

    def __init__(self, uav_type: str, team: str, pos: pygame.math.Vector2, image: pygame.Surface,
                 is_render_mode: bool):  # NEW: is_render_mode param
        self.type = uav_type
        self.team = team  # 'red' or 'blue'
        self.pos = pos
        self.is_alive = True
        self.cooldowns = {  # In combat minutes
            'mw_laser': 0,
            'net': 0
        }
        self.image = image  # This will now always be a pygame.Surface (or DummySurface)

        # --- FIX for AttributeError: 'NoneType' object has no attribute 'get_rect' ---
        # Create self.rect based on render_mode to avoid pygame.Rect issues when not initialized
        if is_render_mode:
            self.rect = image.get_rect(topleft=self.pos)
        else:  # For non-rendering mode, use a simple dummy object that mimics Pygame.Rect
            class SimpleRect:
                def __init__(self, x, y, width, height):
                    self.x = x
                    self.y = y
                    self.width = width
                    self.height = height
                    self.topleft = (x, y)  # Mimic topleft attribute

                # Mimic the ability of Pygame.Rect to have topleft assigned
                def __setattr__(self, name, value):
                    if name == 'topleft' and isinstance(value, (tuple, pygame.math.Vector2)) and len(value) == 2:
                        self.x, self.y = value
                    else:
                        object.__setattr__(self, name, value)

                # Needed if any code tries to do rect.x or rect.y = ...
                @property
                def x(self):
                    return object.__getattribute__(self, '_x')

                @x.setter
                def x(self, value):
                    self._x = value

                @property
                def y(self):
                    return object.__getattribute__(self, '_y')

                @y.setter
                def y(self, value):
                    self._y = value

            # Initialize SimpleRect with current pos and UAV_SIZE
            self.rect = SimpleRect(self.pos.x, self.pos.y, UAV_SIZE, UAV_SIZE)

        self.target_pos = None  # For movement
        self.arrival_threshold = UAV_SPEED * 2  # How close to target_pos to consider "arrived" (2 frames worth of travel)

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
            self.pos = self.target_pos  # Snap to target if very close
            self.target_pos = None  # Arrived
        # Update rect.topleft after pos is updated
        self.rect.topleft = self.pos  # Update rect for drawing (now safe with SimpleRect)

    def reset_cooldowns(self):
        self.cooldowns = {
            'mw_laser': 0,
            'net': 0
        }

    def update_cooldowns(self, delta_time: int = 1):
        """Decrements cooldowns."""
        for weapon_type in self.cooldowns:
            if self.cooldowns[weapon_type] > 0:
                self.cooldowns[weapon_type] -= delta_time

    def can_fire(self, weapon_type: str) -> bool:
        return self.is_alive and self.cooldowns.get(weapon_type, 0) == 0


class UAVCombatEnv:
    """
    A reinforcement learning environment for simulating UAV combat with Pygame visualization.
    """

    def __init__(self,
                 initial_red_uav_counts: Counter = None,  # NEW: Red UAV counts as parameter
                 initial_blue_uav_counts: Counter = None,  # NEW: Blue UAV counts as parameter
                 red_base_rect: pygame.Rect = None,  # NEW: Red base as parameter
                 blue_base_rect: pygame.Rect = None,  # NEW: Blue base as parameter
                 render_mode: bool = True,  # NEW: Render control parameter
                 background_image_path: str = DEFAULT_BACKGROUND_IMAGE,
                 red_interceptor_image_path: str = DEFAULT_RED_INTERCEPTOR_IMAGE,
                 red_recon_image_path: str = DEFAULT_RED_RECON_IMAGE,
                 red_escort_image_path: str = DEFAULT_RED_ESCORT_IMAGE,
                 blue_ground_attack_image_path: str = DEFAULT_BLUE_GROUND_ATTACK_IMAGE,
                 blue_recon_image_path: str = DEFAULT_BLUE_RECON_IMAGE,
                 blue_escort_image_path: str = DEFAULT_BLUE_ESCORT_IMAGE):

        # 1.1 Initial Configuration - Now customizable via parameters
        self.initial_blue_uavs = initial_blue_uav_counts if initial_blue_uav_counts is not None else DEFAULT_INITIAL_BLUE_UAVS.copy()
        self.initial_red_uavs = initial_red_uav_counts if initial_red_uav_counts is not None else DEFAULT_INITIAL_RED_UAVS.copy()

        # Base Rectangles - Now customizable via parameters
        # Use simple tuples/lists for base rects if not rendering
        if render_mode:
            self.RED_BASE_RECT = red_base_rect if red_base_rect is not None else DEFAULT_RED_BASE_RECT
            self.BLUE_BASE_RECT = blue_base_rect if blue_base_rect is not None else DEFAULT_BLUE_BASE_RECT
        else:  # Provide dummy rects that are just (x,y,w,h) tuples for non-render mode
            self.RED_BASE_RECT = (red_base_rect.x, red_base_rect.y, red_base_rect.width,
                                  red_base_rect.height) if red_base_rect is not None else (
            DEFAULT_RED_BASE_RECT.x, DEFAULT_RED_BASE_RECT.y, DEFAULT_RED_BASE_RECT.width, DEFAULT_RED_BASE_RECT.height)
            self.BLUE_BASE_RECT = (blue_base_rect.x, blue_base_rect.y, blue_base_rect.width,
                                   blue_base_rect.height) if blue_base_rect is not None else (
            DEFAULT_BLUE_BASE_RECT.x, DEFAULT_BLUE_BASE_RECT.y, DEFAULT_BLUE_BASE_RECT.width,
            DEFAULT_BLUE_BASE_RECT.height)

        # Render mode control
        self.render_mode = render_mode

        # 1.2 Combat probabilities
        self.PROB_RED_MW_LASER_DMG_BLUE_GA = 0.7
        self.PROB_RED_INTERCEPTOR_NET_DMG_BLUE_GA = 0.9
        self.PROB_BLUE_ESCORT_DMG_RED_INTERCEPTOR_MW_LASER = 0.7

        # Firing intervals (in combat minutes)
        self.INTERVAL_MW_LASER_RED = 1
        self.INTERVAL_MW_LASER_BLUE = 3
        self.INTERVAL_NET_RED = 5  # Red's net interval

        # Current state of Red team's total UAVs
        self.red_uavs = self.initial_red_uavs.copy()
        self.time_step = 0

        # --- Pygame Initialization (only if rendering is enabled) ---
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("UAV Combat Simulation")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)
            self.font_normal = pygame.font.Font(None, FONT_SIZE_NORMAL)
            self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
        else:  # Dummy attributes if not rendering
            self.screen = None
            self.clock = None
            self.font_small = None
            self.font_normal = None
            self.font_large = None

        # Initialize self.images before _load_assets
        self.images = {}
        # Always load assets, _get_uav_image handles dummy creation if not rendering
        self._load_assets(background_image_path, red_interceptor_image_path,
                          red_recon_image_path, red_escort_image_path,
                          blue_ground_attack_image_path, blue_recon_image_path,
                          blue_escort_image_path)

        # Store battlefield coordinates for rendering
        self.current_battlefield_coords = None
        self._battlefield_display_coords = None  # Store integer coords for drawing

        # Lists to hold individual UAV objects deployed in the current combat
        self.red_engaged_uavs_visual = []
        self.blue_engaged_uavs_visual = []

        # For combat animation and log
        self.combat_active_display_time = 0  # Timer for combat summaries
        self.explosion_effects = []  # List of (position, start_time) for explosions
        self.active_attack_lines = []  # List of (start_pos, end_pos, color, end_time)
        self.combat_log = []  # List of recent combat messages (message, display_until_time_ms)

    def _get_uav_image(self, path: str, default_color: tuple, scale_size: tuple) -> pygame.Surface:
        """Helper to load image or create a default colored surface.
           Returns a DummySurface if Pygame is not initialized (render_mode is False)."""

        if not self.render_mode:  # If not rendering, just return a dummy object
            # --- FIX for AttributeError: 'NoneType' object has no attribute 'get_rect' ---
            # Define a simple class that mimics Pygame.Surface and Pygame.Rect attributes
            class DummyRect:
                def __init__(self, topleft, size):
                    self._x, self._y = topleft  # Use internal _x, _y to avoid __setattr__ recursion
                    self.width, self.height = size
                    self._topleft = topleft

                # Properties to mimic Pygame.Rect.x, .y, .topleft behavior for setting
                @property
                def x(self): return self._x

                @x.setter
                def x(self, value): self._x = value

                @property
                def y(self): return self._y

                @y.setter
                def y(self, value): self._y = value

                @property
                def topleft(self): return self._topleft

                @topleft.setter
                def topleft(self, value):
                    self._x, self._y = value
                    self._topleft = value  # Update internal topleft as well

                def __repr__(self):
                    return f"SimpleRect(x={self.x}, y={self.y}, w={self.width}, h={self.height})"

            class DummySurface:
                def __init__(self, size):
                    self._size = size

                def get_rect(self, topleft=(0, 0)):
                    return DummyRect(topleft, self._size)

                def fill(self, *args, **kwargs): pass  # Do nothing

                def copy(self): return self  # For destroyed_image

            return DummySurface(scale_size)  # Instantiate with correct size

        # If rendering, proceed with actual Pygame surface creation
        if path and os.path.exists(path):
            try:
                img = pygame.image.load(path).convert_alpha()
                img = pygame.transform.scale(img, scale_size)
                return img
            except pygame.error as e:
                print(f"Warning: Could not load image from {path}. Error: {e}. Creating default colored rectangle.")

        # Create a default colored rectangle surface
        default_surface = pygame.Surface(scale_size, pygame.SRCALPHA)  # SRCALPHA for transparency
        default_surface.fill(default_color)
        return default_surface

    def _load_assets(self, bg_path, red_interceptor_path, red_recon_path, red_escort_path,
                     blue_ga_path, blue_recon_path, blue_escort_path):
        """Loads images for background and UAVs, with fallbacks to default shapes/colors.
           Always populates self.images, using DummySurface if rendering is off."""

        # Background - set to None if not rendering, otherwise attempt to load
        if self.render_mode:
            if bg_path and os.path.exists(bg_path):
                try:
                    self.images['background'] = pygame.image.load(bg_path).convert_alpha()
                    self.images['background'] = pygame.transform.scale(self.images['background'],
                                                                       (SCREEN_WIDTH, SCREEN_HEIGHT))
                except pygame.error as e:
                    print(f"Warning: Could not load background image from {bg_path}. Error: {e}. Using default color.")
                    self.images['background'] = None
            else:
                self.images['background'] = None
        else:  # If not rendering, background is irrelevant
            self.images['background'] = None

        # UAV images - ALWAYS populate self.images, using _get_uav_image for proper dummy creation
        self.images['red_interceptor'] = self._get_uav_image(red_interceptor_path, RED_COLOR, (UAV_SIZE, UAV_SIZE))
        self.images['red_recon'] = self._get_uav_image(red_recon_path, RED_COLOR, (UAV_SIZE, UAV_SIZE))
        self.images['red_escort'] = self._get_uav_image(red_escort_path, RED_COLOR, (UAV_SIZE, UAV_SIZE))

        self.images['blue_ground_attack'] = self._get_uav_image(blue_ga_path, BLUE_COLOR, (UAV_SIZE, UAV_SIZE))
        self.images['blue_recon'] = self._get_uav_image(blue_recon_path, BLUE_COLOR, (UAV_SIZE, UAV_SIZE))
        self.images['blue_escort'] = self._get_uav_image(blue_escort_path, BLUE_COLOR, (UAV_SIZE, UAV_SIZE))

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self.red_uavs = self.initial_red_uavs.copy()
        self.time_step = 0
        self.combat_active_display_time = 0
        self.red_engaged_uavs_visual = []
        self.blue_engaged_uavs_visual = []
        self.explosion_effects = []
        self.active_attack_lines = []
        self.combat_log = []
        self.current_battlefield_coords = None  # Clear battlefield for reset
        self._battlefield_display_coords = None  # Clear display coords on reset

        initial_state = {
            'friendly_remaining': {
                'interceptor': self.red_uavs['interceptor'],
                'recon': self.red_uavs['recon'],
                'escort': self.red_uavs['escort']
            }
        }
        if self.render_mode:
            self.render(initial_state)  # Render initial state (bases only)
        return initial_state

    def _calculate_damage(self, attacker_count: int, damage_prob: float) -> int:
        """
        Calculates the number of successful hits based on attacker count and damage probability.
        """
        hits = 0
        for _ in range(attacker_count):
            if np.random.rand() < damage_prob:
                hits += 1
        return hits

    def step(self, action: dict, enemy_formation: dict, battlefield_coords: tuple):
        """
        Performs one step in the environment, simulating a multi-minute combat.
        Args:
            action (dict): Friendly UAVs to deploy.
            enemy_formation (dict): Enemy UAVs in the current cluster.
            battlefield_coords (tuple): (x, y) coordinates of the center of the battlefield.
        """
        self.current_battlefield_coords = pygame.math.Vector2(battlefield_coords)
        # Store integer coordinates for fixed drawing position
        self._battlefield_display_coords = (
        int(self.current_battlefield_coords.x), int(self.current_battlefield_coords.y))
        # --- Apply Deployment Constraints ---
        # Red Team Constraint: At least 1侦察无人机, 1拦截机
        initial_action_recon = action.get('recon', 0)
        initial_action_interceptor = action.get('interceptor', 0)

        if initial_action_recon < 1 and self.red_uavs['recon'] >= 1:
            print(
                f"Warning: Red action does not meet '1 Recon' constraint (current: {initial_action_recon}). Adjusting action to 1.")
            action['recon'] = 1
        if initial_action_interceptor < 1 and self.red_uavs['interceptor'] >= 1:
            print(
                f"Warning: Red action does not meet '1 Interceptor' constraint (current: {initial_action_interceptor}). Adjusting action to 1.")
            action['interceptor'] = 1

        # Blue Team Constraint: At least 1侦察无人机, 1对地攻击无人机
        initial_enemy_recon = enemy_formation.get('recon', 0)
        initial_enemy_ga = enemy_formation.get('ground_attack', 0)

        if initial_enemy_recon < 1:
            print(
                f"Warning: Enemy formation does not meet '1 Recon' constraint (current: {initial_enemy_recon}). Adjusting formation to 1.")
            enemy_formation['recon'] = 1
        if initial_enemy_ga < 1:
            print(
                f"Warning: Enemy formation does not meet '1 Ground Attack' constraint (current: {initial_enemy_ga}). Adjusting formation to 1.")
            enemy_formation['ground_attack'] = 1

        # Ensure deployed friendly UAVs do not exceed available ones (after potential constraint adjustments)
        action_interceptor = min(action.get('interceptor', 0), self.red_uavs['interceptor'])
        action_recon = min(action.get('recon', 0), self.red_uavs['recon'])
        action_escort = min(action.get('escort', 0), self.red_uavs['escort'])

        deployed_friendly = Counter({
            'interceptor': action_interceptor,
            'recon': action_recon,
            'escort': action_escort
        })
        current_enemy_formation = Counter(enemy_formation)  # Use the potentially adjusted enemy_formation

        # Clear previous engagement's visual UAVs
        self.red_engaged_uavs_visual = []
        self.blue_engaged_uavs_visual = []
        self.explosion_effects = []
        self.active_attack_lines = []
        self.combat_log = []

        # --- Populate visual UAVs for this specific engagement ---
        # Red side: Start positions within Red's base
        for uav_type, count in deployed_friendly.items():
            image_surface = self.images.get(f'red_{uav_type}')
            for _ in range(count):
                # Random position within the base rectangle
                x = np.random.uniform(self.RED_BASE_RECT[0], self.RED_BASE_RECT[0] + self.RED_BASE_RECT[2] - UAV_SIZE)
                y = np.random.uniform(self.RED_BASE_RECT[1], self.RED_BASE_RECT[1] + self.RED_BASE_RECT[3] - UAV_SIZE)
                self.red_engaged_uavs_visual.append(
                    UAV(uav_type, 'red', pygame.math.Vector2(x, y), image_surface, self.render_mode)  # Pass render_mode
                )

        # Blue side: Start positions within Blue's base
        for uav_type, count in current_enemy_formation.items():
            if uav_type not in ['ground_attack', 'recon', 'escort']:  # Only allowed Blue types
                print(f"Warning: Blue formation contains unsupported UAV type '{uav_type}'. Skipping.")
                continue

            image_surface = self.images.get(f'blue_{uav_type}')
            for _ in range(count):
                # Random position within the base rectangle
                x = np.random.uniform(self.BLUE_BASE_RECT[0],
                                      self.BLUE_BASE_RECT[0] + self.BLUE_BASE_RECT[2] - UAV_SIZE)
                y = np.random.uniform(self.BLUE_BASE_RECT[1],
                                      self.BLUE_BASE_RECT[1] + self.BLUE_BASE_RECT[3] - UAV_SIZE)
                self.blue_engaged_uavs_visual.append(
                    UAV(uav_type, 'blue', pygame.math.Vector2(x, y), image_surface, self.render_mode)
                    # Pass render_mode
                )

        # Set target for all engaged UAVs to the battlefield center
        for uav in self.blue_engaged_uavs_visual:
            x, y = battlefield_coords
            x = x + 20
            uav.set_target_pos(pygame.math.Vector2((x, y)))

        for uav in self.red_engaged_uavs_visual:
            x, y = battlefield_coords
            x = x - 20
            uav.set_target_pos(pygame.math.Vector2((x, y)))

        # --- Combined Travel & Combat Phase ---
        # Total frames for the combined travel and combat
        total_simulation_frames = COMBAT_SIM_MINUTES * FRAMES_PER_COMBAT_MINUTE

        cumulative_red_losses = Counter()
        cumulative_blue_losses = Counter()

        # Helper function to apply hits and log events
        def apply_hits_and_log(firing_units_list, target_candidates_list, damage_prob, attacker_type_str, weapon_name,
                               is_red_attacker=True):
            hits = self._calculate_damage(len(firing_units_list), damage_prob)

            destroyed_count = 0
            if target_candidates_list:
                # Shuffle targets to pick randomly from the available pool each time
                np.random.shuffle(target_candidates_list)

                for i in range(min(hits, len(target_candidates_list))):
                    target = target_candidates_list[i]
                    if target.is_alive:  # Double check if not already destroyed this minute
                        target.is_alive = False
                        self.explosion_effects.append((target.pos.copy(), pygame.time.get_ticks()))

                        if is_red_attacker:
                            cumulative_blue_losses[target.type] += 1
                            self.combat_log.append((
                                                   f"Min {current_combat_minute}: {attacker_type_str} {weapon_name} destroyed Blue {target.type.replace('_', ' ').title()}!",
                                                   pygame.time.get_ticks() + LOG_DISPLAY_TIME_MS))
                        else:
                            cumulative_red_losses[target.type] += 1
                            self.combat_log.append((
                                                   f"Min {current_combat_minute}: Blue {attacker_type_str} {weapon_name} destroyed Red {target.type.replace('_', ' ').title()}!",
                                                   pygame.time.get_ticks() + LOG_DISPLAY_TIME_MS))

                        # Add attack line visual (from a random attacker to target)
                        if self.render_mode and firing_units_list:  # Only add visual effect if rendering
                            attacker_uav_for_visual = np.random.choice(firing_units_list)
                            line_color = (255, 255, 0) if "MW/Laser" in weapon_name else (
                            255, 0, 255)  # Yellow for laser, Magenta for net
                            self.active_attack_lines.append((attacker_uav_for_visual.pos + pygame.math.Vector2(
                                UAV_SIZE / 2, UAV_SIZE / 2), target.pos + pygame.math.Vector2(UAV_SIZE / 2,
                                                                                              UAV_SIZE / 2), line_color,
                                                             pygame.time.get_ticks() + ATTACK_EFFECT_DISPLAY_TIME_MS))

                        destroyed_count += 1

            # Set cooldown for the firing units
            cooldown_key = weapon_name.lower().split(' ')[0]  # 'mw/laser' -> 'mw', 'net' -> 'net'
            if '/' in cooldown_key: cooldown_key = 'mw_laser'  # Handle mw/laser
            for uav in firing_units_list:
                uav.cooldowns[
                    cooldown_key] = self.INTERVAL_MW_LASER_RED if is_red_attacker else self.INTERVAL_MW_LASER_BLUE if cooldown_key == 'mw_laser' else self.INTERVAL_NET_RED

        for frame_num in range(total_simulation_frames):  # Loop for entire simulation duration
            current_combat_minute = frame_num // FRAMES_PER_COMBAT_MINUTE + 1

            # UAVs continue to move towards battlefield center until they arrive
            for uav in self.red_engaged_uavs_visual + self.blue_engaged_uavs_visual:
                uav.move_towards(UAV_SPEED)

            # --- Weapon Firing Logic (only at the start of relevant minutes) ---
            if frame_num % FRAMES_PER_COMBAT_MINUTE == 0:
                # Filter units that are alive AND within engagement range of battlefield center
                active_red_interceptors = [u for u in self.red_engaged_uavs_visual if
                                           u.is_alive and u.type == 'interceptor' and u.pos.distance_to(
                                               self.current_battlefield_coords) <= UAV_ENGAGEMENT_RADIUS]
                active_red_escorts = [u for u in self.red_engaged_uavs_visual if
                                      u.is_alive and u.type == 'escort' and u.pos.distance_to(
                                          self.current_battlefield_coords) <= UAV_ENGAGEMENT_RADIUS]

                active_blue_ground_attack = [u for u in self.blue_engaged_uavs_visual if
                                             u.is_alive and u.type == 'ground_attack' and u.pos.distance_to(
                                                 self.current_battlefield_coords) <= UAV_ENGAGEMENT_RADIUS]
                active_blue_escorts = [u for u in self.blue_engaged_uavs_visual if
                                       u.is_alive and u.type == 'escort' and u.pos.distance_to(
                                           self.current_battlefield_coords) <= UAV_ENGAGEMENT_RADIUS]

                active_blue_flys = [u for u in self.blue_engaged_uavs_visual if
                                       u.is_alive and u.pos.distance_to(
                                           self.current_battlefield_coords) <= UAV_ENGAGEMENT_RADIUS]
                active_red_flys = [u for u in self.red_engaged_uavs_visual if
                                      u.is_alive and u.pos.distance_to(
                                          self.current_battlefield_coords) <= UAV_ENGAGEMENT_RADIUS]

                # Red MW/Laser attack (Interceptors & Escorts) on Blue Ground Attack
                if current_combat_minute % self.INTERVAL_MW_LASER_RED == 0:
                    firing_units = [u for u in active_red_interceptors + active_red_escorts if u.can_fire('mw_laser')]
                    apply_hits_and_log(firing_units, active_blue_flys, self.PROB_RED_MW_LASER_DMG_BLUE_GA,
                                       "Red (Int/Esc)", "MW/Laser", is_red_attacker=True)

                # Red Interceptor Net attack (every 5 minutes) on Blue Ground Attack
                if current_combat_minute % self.INTERVAL_NET_RED == 0:
                    firing_units = [u for u in active_red_interceptors if u.can_fire('net')]
                    apply_hits_and_log(firing_units, active_blue_flys,
                                       self.PROB_RED_INTERCEPTOR_NET_DMG_BLUE_GA, "Red Interceptor", "Net",
                                       is_red_attacker=True)

                # Blue Escorts fire Microwave/Laser (every 3 minutes) on Red Interceptors
                if current_combat_minute % self.INTERVAL_MW_LASER_BLUE == 0:
                    firing_units = [u for u in active_blue_escorts if u.can_fire('mw_laser')]
                    apply_hits_and_log(firing_units, active_red_flys,
                                       self.PROB_BLUE_ESCORT_DMG_RED_INTERCEPTOR_MW_LASER, "Escort", "MW/Laser",
                                       is_red_attacker=False)

            # Decrement cooldowns for all engaged UAVs (only when a new minute starts)
            if frame_num % FRAMES_PER_COMBAT_MINUTE == 0:
                for uav in self.red_engaged_uavs_visual + self.blue_engaged_uavs_visual:
                    uav.update_cooldowns(delta_time=1)

            if self.render_mode:  # Only render if enabled
                self.render(None, None)  # Render current frame
                self.clock.tick(FPS)

                # --- Early Combat Termination Check (NEW) ---
            active_red_engaged_in_loop = [u for u in self.red_engaged_uavs_visual if u.is_alive]
            active_blue_engaged_in_loop = [u for u in self.blue_engaged_uavs_visual if u.is_alive]

            if not active_red_engaged_in_loop or not active_blue_engaged_in_loop:
                # One side's engaged forces have been wiped out! Break the combat loop early.
                current_time = pygame.time.get_ticks()  # Get current time for log
                self.combat_log.append((
                                       f"Min {current_combat_minute}: Engaged forces of one side wiped out! Combat ends early.",
                                       current_time + LOG_DISPLAY_TIME_MS * 2))
                if self.render_mode:  # Ensure summary is displayed if rendering
                    self.combat_active_display_time = FPS * 2  # Display for 2 seconds
                    self.render(None, None)  # Render the final state of this combat
                    pygame.time.wait(2000)  # Wait for display
                break  # Exit the total_simulation_frames loop

        # --- After combat simulation (total_simulation_frames completed or early break) ---
        self.red_uavs.subtract(action)
        self.initial_blue_uavs.subtract(enemy_formation)

        for key in self.red_uavs:
            if self.red_uavs[key] < 0:
                self.red_uavs[key] = 0
        reward = 0   # reward = 3 * num_ground_attacks + 1 * escort + 1 * recon - 0.5 + red_losses
        for item, value in cumulative_blue_losses.items():
            if item == 'ground_attack':
                reward += value * 3
            else:
                reward += value

        reward -= sum(cumulative_red_losses.values()) * 0.5

        next_state = {
            'friendly_remaining': {
                'interceptor': self.red_uavs['interceptor'],
                'recon': self.red_uavs['recon'],
                'escort': self.red_uavs['escort']
            },
            'current_enemy_formation_remaining': {k: v for k, v in self.initial_blue_uavs.items() if v > 0}
        }

        # Done condition: if Red's primary combat units (interceptors or escorts) are depleted, or all UAVs are gone.
        # This 'done' is for the overall RL episode.
        done = (self.red_uavs['interceptor'] <= 0 and self.red_uavs['escort'] <= 0) or (
                    sum(self.red_uavs.values()) <= 0)
        self.time_step += 1

        info = {
            'red_losses_in_step': cumulative_red_losses,
            'blue_losses_in_step': cumulative_blue_losses,
            'deployed_friendly': deployed_friendly
        }

        if self.render_mode:  # Only render if enabled
            # This final summary display will only happen if the combat loop didn't break early
            # or if it broke early, the `if self.render_mode:` block above already handled a wait.
            if self.combat_active_display_time > 0:  # Check if summary is still needed to be shown
                self.render(next_state, info)  # Final render of summary
                pygame.time.wait(2000)  # Wait 2 seconds before closing window automatically if done

        return next_state, reward, done, info

    def render(self, state: dict = None, info: dict = None):
        """
        Renders the current state of the environment using Pygame.
        Called frequently during combat sub-loop and once at reset/step end.
        """
        if not self.render_mode: return  # Skip rendering if not enabled

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw background
        if self.images['background']:
            self.screen.blit(self.images['background'], (0, 0))
        else:
            self.screen.fill(GREY_COLOR)

        # --- Draw Bases ---
        # Access tuple elements directly for base rects when not rendering
        pygame.draw.rect(self.screen, RED_COLOR, self.RED_BASE_RECT, 2)  # Red base outline
        red_base_text = self.font_small.render("Red Base", True, RED_COLOR)
        self.screen.blit(red_base_text,
                         (self.RED_BASE_RECT[0] + 5, self.RED_BASE_RECT[1] + 5))  # Use [0] and [1] for x,y

        pygame.draw.rect(self.screen, BLUE_COLOR, self.BLUE_BASE_RECT, 2)  # Blue base outline
        blue_base_text = self.font_small.render("Blue Base", True, BLUE_COLOR)
        self.screen.blit(blue_base_text,
                         (self.BLUE_BASE_RECT[0] + 5, self.BLUE_BASE_RECT[1] + 5))  # Use [0] and [1] for x,y

        # --- Draw Battlefield Marker ---
        if self.current_battlefield_coords:
            # Use the stored integer coordinates for pixel-perfect drawing
            pygame.draw.circle(self.screen, BATTLEFIELD_COLOR, self._battlefield_display_coords,
                               BATTLEFIELD_MARKER_RADIUS, 2)
            battlefield_text = self.font_small.render("Battlefield", True, BATTLEFIELD_COLOR)
            self.screen.blit(battlefield_text, (self._battlefield_display_coords[0] - battlefield_text.get_width() / 2,
                                                self._battlefield_display_coords[
                                                    1] - BATTLEFIELD_MARKER_RADIUS - battlefield_text.get_height() - 5))
            # Also draw the larger engagement radius for visual clarity
            pygame.draw.circle(self.screen, (BATTLEFIELD_COLOR[0], BATTLEFIELD_COLOR[1], BATTLEFIELD_COLOR[2], 50),
                               self._battlefield_display_coords, UAV_ENGAGEMENT_RADIUS, 1)

        # --- Display Total Friendly (Red) UAVs ---
        # Only display if a full state is provided (e.g., outside combat loop, or at the end of a step)
        if state:
            red_interceptor_count = state['friendly_remaining']['interceptor']
            red_recon_count = state['friendly_remaining']['recon']
            red_escort_count = state['friendly_remaining']['escort']

            text_red_interceptor = self.font_normal.render(f"Red Interceptors (Total): {red_interceptor_count}", True,
                                                           RED_COLOR)
            text_red_recon = self.font_normal.render(f"Red Recon (Total): {red_recon_count}", True, RED_COLOR)
            text_red_escort = self.font_normal.render(f"Red Escort (Total): {red_escort_count}", True, RED_COLOR)

            self.screen.blit(text_red_interceptor, (20, 20))
            self.screen.blit(text_red_recon, (20, 50))
            self.screen.blit(text_red_escort, (20, 80))

            # Display remaining enemy formation counts from the current state (if applicable)
            enemy_formation_remaining = state.get('current_enemy_formation_remaining', {})

            text_blue_engaged = self.font_normal.render(f"Blue Engaged (remaining):", True, BLUE_COLOR)
            self.screen.blit(text_blue_engaged, (SCREEN_WIDTH - text_blue_engaged.get_width() - 20, 20))

            y_offset_blue_text = 50
            for uav_type, count in enemy_formation_remaining.items():
                text_type_count = self.font_normal.render(f"  {uav_type.replace('_', ' ').title()}: {count}", True,
                                                          BLUE_COLOR)
                self.screen.blit(text_type_count, (SCREEN_WIDTH - text_type_count.get_width() - 20, y_offset_blue_text))
                y_offset_blue_text += 30

        # --- Draw Individual Engaged UAVs ---
        for uav in self.red_engaged_uavs_visual:
            if uav.is_alive:
                self.screen.blit(uav.image, uav.pos)
            else:  # Draw destroyed unit
                destroyed_image = uav.image.copy()
                destroyed_image.fill(DESTROYED_COLOR + (0,), special_flags=pygame.BLEND_RGBA_MULT)
                destroyed_image.fill((0, 0, 0, 100), special_flags=pygame.BLEND_RGBA_ADD)
                self.screen.blit(destroyed_image, uav.pos)
                pygame.draw.line(self.screen, BLACK, uav.pos, (uav.pos.x + UAV_SIZE, uav.pos.y + UAV_SIZE), 2)
                pygame.draw.line(self.screen, BLACK, (uav.pos.x + UAV_SIZE, uav.pos.y),
                                 (uav.pos.x, uav.pos.y + UAV_SIZE), 2)

        for uav in self.blue_engaged_uavs_visual:
            if uav.is_alive:
                self.screen.blit(uav.image, uav.pos)
            else:  # Draw destroyed unit
                destroyed_image = uav.image.copy()
                destroyed_image.fill(DESTROYED_COLOR + (0,), special_flags=pygame.BLEND_RGBA_MULT)
                destroyed_image.fill((0, 0, 0, 100), special_flags=pygame.BLEND_RGBA_ADD)
                self.screen.blit(destroyed_image, uav.pos)
                pygame.draw.line(self.screen, BLACK, uav.pos, (uav.pos.x + UAV_SIZE, uav.pos.y + UAV_SIZE), 2)
                pygame.draw.line(self.screen, BLACK, (uav.pos.x + UAV_SIZE, uav.pos.y),
                                 (uav.pos.x, uav.pos.y + UAV_SIZE), 2)

        # --- Draw Explosion Effects ---
        current_time = pygame.time.get_ticks()
        active_explosions = []
        for pos, start_time in self.explosion_effects:
            if current_time - start_time < 500:  # Show for 0.5 seconds
                explosion_radius = (current_time - start_time) / 500 * (UAV_SIZE * 1.5)
                pygame.draw.circle(self.screen, (255, 165, 0), pos + pygame.math.Vector2(UAV_SIZE / 2, UAV_SIZE / 2),
                                   int(explosion_radius), 2)
                active_explosions.append((pos, start_time))
        self.explosion_effects = active_explosions

        # --- Draw Active Attack Lines ---
        active_lines = []
        for start_pos, end_pos, color, end_time in self.active_attack_lines:
            if current_time < end_time:
                pygame.draw.line(self.screen, color, start_pos, end_pos, 3)  # Draw thicker line
                active_lines.append((start_pos, end_pos, color, end_time))
        self.active_attack_lines = active_lines

        # --- Combat Log Display ---
        y_log_start = SCREEN_HEIGHT - 100
        # Sort log to keep newest at bottom, but display order is from top
        display_log = sorted(self.combat_log, key=lambda x: x[1])[-MAX_LOG_MESSAGES:]
        for i, (msg, display_until_time) in enumerate(display_log):
            if current_time < display_until_time:
                log_text = self.font_small.render(msg, True, WHITE)
                self.screen.blit(log_text, (20, y_log_start + i * (FONT_SIZE_SMALL + 2)))
        self.combat_log = [(msg, dt) for msg, dt in self.combat_log if current_time < dt]

        # --- Display Time Step and Reward Summary ---
        time_text = self.font_normal.render(f"Global Time Step: {self.time_step}", True, BLACK)
        self.screen.blit(time_text, (SCREEN_WIDTH // 2 - time_text.get_width() // 2, 20))

        if self.combat_active_display_time > 0 and info:
            combat_summary_text = self.font_large.render("--- Engagement Summary ---", True, WHITE)
            red_losses_text = self.font_large.render(f"Red Losses: {sum(info['red_losses_in_step'].values())}", True,
                                                     WHITE)
            blue_losses_text = self.font_large.render(f"Blue Destroyed: {sum(info['blue_losses_in_step'].values())}",
                                                      True, WHITE)

            self.screen.blit(combat_summary_text,
                             (SCREEN_WIDTH // 2 - combat_summary_text.get_width() // 2, SCREEN_HEIGHT // 2 - 80))
            self.screen.blit(red_losses_text,
                             (SCREEN_WIDTH // 2 - red_losses_text.get_width() // 2, SCREEN_HEIGHT // 2 - 20))
            self.screen.blit(blue_losses_text,
                             (SCREEN_WIDTH // 2 - blue_losses_text.get_width() // 2, SCREEN_HEIGHT // 2 + 40))

            self.combat_active_display_time -= 1

        pygame.display.flip()

    def close(self):
        """Closes the Pygame window."""
        if self.render_mode:  # Only quit pygame if it was initialized
            pygame.quit()


# --- Example Usage ---
if __name__ == '__main__':
    # You can specify image paths here. If they don't exist or are None,
    # default colored rectangles will be used.
    # Create an 'assets' folder in the same directory as this script
    # and place your images there.
    # Recommended image names for full visual support:
    # 'assets/background.png'
    # 'assets/red_interceptor.png'
    # 'assets/red_recon.png'
    # 'assets/red_escort.png'
    # 'assets/blue_ground_attack.png'
    # 'assets/blue_recon.png'
    # 'assets/blue_escort.png'

    # --- Custom Initial Configurations ---
    my_red_uavs = Counter(
        {'interceptor': 40, 'recon': 10, 'escort': 0})  # Example: More interceptors, fewer recon/escort
    my_blue_uavs = Counter({'ground_attack': 10, 'recon': 10, 'escort': 30})  # Example: More ground attack

    # Custom Base Rectangles (x, y, width, height)
    my_red_base = pygame.Rect(20, SCREEN_HEIGHT / 2 - 100, 150, 200)
    my_blue_base = pygame.Rect(SCREEN_WIDTH - 170, SCREEN_HEIGHT / 2 - 100, 150, 200)

    # --- Scenario 1: With Rendering ---
    print("--- Running Scenario 1: With Rendering Enabled ---")
    env_render = UAVCombatEnv(
        initial_red_uav_counts=my_red_uavs,
        initial_blue_uav_counts=my_blue_uavs,
        red_base_rect=my_red_base,
        blue_base_rect=my_blue_base,
        render_mode=True,  # Enable rendering
        background_image_path='figure/背景.png',
        red_interceptor_image_path='figure/红.png',
        red_recon_image_path='figure/红3.png',
        red_escort_image_path='figure/红2.png',
        blue_ground_attack_image_path='figure/蓝.png',
        blue_recon_image_path='figure/蓝3.png',
        blue_escort_image_path='figure/蓝2.png'
    )

    state = env_render.reset()
    done = False
    total_reward = 0
    step_count = 0
    max_global_steps = 3  # Run fewer steps for demo with rendering

    scenario_sequence_render = [
        {'enemy_formation': {'ground_attack': 2, 'recon': 1, 'escort': 3},
         'battlefield_coords': (SCREEN_WIDTH * 0.4, SCREEN_HEIGHT * 0.5)},
        {'enemy_formation': {'ground_attack': 1, 'recon': 1, 'escort': 2},
         'battlefield_coords': (SCREEN_WIDTH * 0.6, SCREEN_HEIGHT * 0.3)},
        {'enemy_formation': {'ground_attack': 4, 'recon': 2, 'escort': 5},
         'battlefield_coords': (SCREEN_WIDTH * 0.5, SCREEN_HEIGHT * 0.7)},
    ]

    while not done and step_count < max_global_steps:
        action = {'interceptor': 10, 'recon': 1, 'escort': 0}

        if step_count < len(scenario_sequence_render):
            current_scenario = scenario_sequence_render[step_count]
            current_enemy_formation = current_scenario['enemy_formation']
            current_battlefield_coords = current_scenario['battlefield_coords']
        else:
            current_enemy_formation = {
                'ground_attack': np.random.randint(1, 4), 'recon': np.random.randint(1, 3),
                'escort': np.random.randint(0, 4)
            }
            current_battlefield_coords = (np.random.uniform(SCREEN_WIDTH * 0.3, SCREEN_WIDTH * 0.7),
                                          np.random.uniform(SCREEN_HEIGHT * 0.3, SCREEN_HEIGHT * 0.7))

        next_state, reward, done, info = env_render.step(action=action, enemy_formation=current_enemy_formation,
                                                         battlefield_coords=current_battlefield_coords)
        total_reward += reward
        step_count += 1

        print(f"Rendered Step {step_count}: Reward={reward}, Done={done}, Red Left={next_state['friendly_remaining']}")
        pygame.time.wait(2000)  # Pause for 2 seconds after each step summary

    env_render.close()
    print(f"Scenario 1 Ended. Final Total Reward: {total_reward}\n")

    # --- Scenario 2: Without Rendering (for faster simulation) ---
    print("--- Running Scenario 2: With Rendering Disabled (Faster Simulation) ---")
    env_no_render = UAVCombatEnv(
        initial_red_uav_counts=my_red_uavs,
        initial_blue_uav_counts=my_blue_uavs,
        red_base_rect=my_red_base,
        blue_base_rect=my_blue_base,
        render_mode=False  # Disable rendering
    )

    state = env_no_render.reset()
    done = False
    total_reward = 0
    step_count = 0
    max_global_steps = 100  # Can run many more steps quickly without rendering

    # Use a different sequence or random generation for non-rendered scenario
    while not done and step_count < max_global_steps:
        action = {'interceptor': 10, 'recon': 1, 'escort': 0}
        current_enemy_formation = {
            'ground_attack': np.random.randint(1, 4), 'recon': np.random.randint(1, 3),
            'escort': np.random.randint(0, 4)
        }
        current_battlefield_coords = (np.random.uniform(SCREEN_WIDTH * 0.3, SCREEN_WIDTH * 0.7),
                                      np.random.uniform(SCREEN_HEIGHT * 0.3, SCREEN_HEIGHT * 0.7))

        next_state, reward, done, info = env_no_render.step(action=action, enemy_formation=current_enemy_formation,
                                                            battlefield_coords=current_battlefield_coords)
        total_reward += reward
        step_count += 1

        # if step_count % 10 == 0:  # Print less frequently for non-rendered mode
        print(
            f"Non-Rendered Step {step_count}: Reward={reward}, Done={done}, Red Left={next_state['friendly_remaining']}")

    env_no_render.close()  # Still good practice to call close even if Pygame wasn't initialized, it handles it.
    print(f"Scenario 2 Ended. Final Total Reward: {total_reward}")