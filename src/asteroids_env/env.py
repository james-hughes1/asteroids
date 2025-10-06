import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import random

class AsteroidsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode="rgb_array", width=128, height=128, max_steps=1000, num_asteroids=5, max_asteroid_size=90, max_asteroid_speed=0.5):
        super().__init__()

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.num_asteroids = num_asteroids
        self.max_asteroid_size = max_asteroid_size
        self.max_asteroid_speed = max_asteroid_speed
        self.render_mode = render_mode

        # Discrete actions: nothing, rotate left, rotate right, thrust, shoot
        self.action_space = spaces.Discrete(5)

        # Observation is image of the game
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        # Pygame surface for rendering
        self.screen = None
        self.clock = None

        self._init_game()

    def _init_game(self):
        # Ship
        self.ship_width = 10
        self.ship_height = 35
        self.ship_x = self.width / 2
        self.ship_y = self.height / 2
        self.ship_angle = 0
        self.ship_speed = 0
        self.ship_max_speed = 5

        # Bullets: list of [x, y, dx, dy]
        self.bullets = []
        self.bullet_cooldown = 0

        # Asteroids: list of [x, y, dx, dy, size]
        self.asteroids = []
        for _ in range(self.num_asteroids):
            x, y = random.randint(0, self.width), random.randint(0, self.height)
            while abs(x - self.ship_x) < 100 and abs(y - self.ship_y) < 100:
                x, y = random.randint(0, self.width), random.randint(0, self.height)
            dx, dy = random.uniform(-self.max_asteroid_speed, self.max_asteroid_speed), random.uniform(-self.max_asteroid_speed, self.max_asteroid_speed)
            size = random.randint(30, self.max_asteroid_size)
            self.asteroids.append([x, y, dx, dy, size])

        self.steps = 0
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # ensures Gym handles seeding
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._init_game()
        return self._get_obs(), {}
    
    def _get_masks(self, nearby_asteroids):
        """Return binary masks for ship and only nearby asteroids for pixel-perfect collision detection."""
        # --- Ship mask ---
        ship_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        ship_surf.fill((0, 0, 0, 0))  # transparent

        # Draw triangle ship in white
        ship_points = [
            (self.ship_x, self.ship_y - self.ship_height / 2),
            (self.ship_x - self.ship_width, self.ship_y + self.ship_height / 2),
            (self.ship_x + self.ship_width, self.ship_y + self.ship_height / 2)
        ]
        theta = math.radians(self.ship_angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rotated_points = []
        for x, y in ship_points:
            dx, dy = x - self.ship_x, y - self.ship_y
            rx = dx * cos_t + dy * sin_t + self.ship_x
            ry = - dx * sin_t + dy * cos_t + self.ship_y
            rotated_points.append((int(rx), int(ry)))

        pygame.draw.polygon(ship_surf, (255, 255, 255), rotated_points)
        ship_mask = pygame.surfarray.array3d(ship_surf).max(axis=2) > 0

        # --- Asteroid masks (only for nearby ones) ---
        asteroid_masks = []
        for a in nearby_asteroids:
            ax, ay, _, _, size = a
            ast_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            ast_surf.fill((0, 0, 0, 0))
            pygame.draw.circle(ast_surf, (255, 255, 255), (int(ax), int(ay)), int(size / 2))
            ast_mask = pygame.surfarray.array3d(ast_surf).max(axis=2) > 0
            asteroid_masks.append(ast_mask)

        return ship_mask, asteroid_masks

    def step(self, action):
        reward = 0
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # --- Apply action ---
        rotation_speed = 8
        acceleration = 0
        bullet_speed = 5

        if action == 1:  # rotate left
            self.ship_angle += rotation_speed
        elif action == 2:  # rotate right
            self.ship_angle -= rotation_speed
        elif action == 3:  # thrust
            acceleration = 0.2
        elif action == 4:  # shoot
            if self.bullet_cooldown == 0:
                if len(self.bullets) < 5:
                    self.bullet_cooldown = 10  # frames until next shot
                    rad = math.radians(self.ship_angle)
                    bx = self.ship_x + -math.sin(rad) * self.ship_height / 2
                    by = self.ship_y + -math.cos(rad) * self.ship_height / 2
                    dx = -math.sin(rad) * bullet_speed
                    dy = -math.cos(rad) * bullet_speed
                    self.bullets.append([bx, by, dx, dy])
            else:
                self.bullet_cooldown -= 1

        # friction
        self.ship_speed *= 0.98
        rad = math.radians(self.ship_angle)
        self.ship_speed = min(self.ship_speed + acceleration, self.ship_max_speed)
        self.ship_x += -math.sin(rad) * self.ship_speed
        self.ship_y += -math.cos(rad) * self.ship_speed

        # wrap-around
        self.ship_x %= self.width
        self.ship_y %= self.height

        # --- Update bullets ---
        for b in self.bullets:
            b[0] += b[2]
            b[1] += b[3]
        self.bullets = [b for b in self.bullets if 0 <= b[0] <= self.width and 0 <= b[1] <= self.height]

        # --- Update asteroids ---
        for a in self.asteroids:
            a[0] += a[2]
            a[1] += a[3]
            a[0] %= self.width
            a[1] %= self.height

        # --- Collision detection bullets vs asteroids ---
        new_asteroids = []
        for a in self.asteroids:
            ax, ay, adx, ady, size = a
            hit = False
            for b in self.bullets:
                if (ax - b[0]) ** 2 + (ay - b[1]) ** 2 < (size / 2) ** 2:
                    hit = True
                    self.bullets.remove(b)
                    reward += (200 - size) / 200  # smaller asteroids give more reward
                    if size > 30:
                        # split asteroid
                        for _ in range(2):
                            ndx, ndy = random.uniform(-self.max_asteroid_speed, self.max_asteroid_speed), random.uniform(-self.max_asteroid_speed, self.max_asteroid_speed)
                            new_asteroids.append([ax, ay, ndx, ndy, size // 2])
                    break
            if not hit:
                new_asteroids.append(a)
        self.asteroids = new_asteroids

        # --- Ship vs asteroids (broad-phase + precise check) ---
        ship_radius = max(self.ship_width, self.ship_height)
        nearby_asteroids = []
        for a in self.asteroids:
            ax, ay, _, _, size = a
            if math.hypot(ax - self.ship_x, ay - self.ship_y) < (size / 2 + ship_radius):
                nearby_asteroids.append(a)

        if nearby_asteroids:
            ship_mask, asteroid_masks = self._get_masks(nearby_asteroids)
            for ast_mask in asteroid_masks:
                if np.any(ship_mask & ast_mask):
                    reward -= 1
                    self.done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        # return image of game
        surface = pygame.Surface((self.width, self.height))
        surface.fill((0, 0, 0))

        # draw asteroids
        for a in self.asteroids:
            pygame.draw.circle(surface, (255, 0, 0), (int(a[0]), int(a[1])), a[4] // 2, 4)

        # draw bullets
        for b in self.bullets:
            pygame.draw.circle(surface, (0, 255, 0), (int(b[0]), int(b[1])), 2)

        # draw ship (thin triangle)
        ship_surf = pygame.Surface((self.ship_width*2, self.ship_height), pygame.SRCALPHA)
        pygame.draw.polygon(
            ship_surf,
            (0, 0, 255),
            [(self.ship_width, 0), (0, self.ship_height), (self.ship_width*2, self.ship_height)],
        )
        rotated_ship = pygame.transform.rotate(ship_surf, self.ship_angle)
        rect = rotated_ship.get_rect(center=(self.ship_x, self.ship_y))
        surface.blit(rotated_ship, rect.topleft)

        if self.render_mode == "rgb_array":
            return np.array(pygame.surfarray.array3d(surface)).transpose(1,0,2)
        elif self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                self.clock = pygame.time.Clock()
            self.screen.blit(surface, (0,0))
            pygame.display.flip()
            self.clock.tick(60)
            return np.array(pygame.surfarray.array3d(surface)).transpose(1,0,2)
