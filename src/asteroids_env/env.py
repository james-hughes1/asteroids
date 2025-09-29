import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import random

class AsteroidsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode="rgb_array", width=128, height=128, max_steps=1000):
        super().__init__()

        self.width = width
        self.height = height
        self.max_steps = max_steps
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
        self.ship_max_speed = 4

        # Bullets: list of [x, y, dx, dy]
        self.bullets = []
        self.bullet_cooldown = 0

        # Asteroids: list of [x, y, dx, dy, size]
        self.asteroids = []
        for _ in range(5):
            x, y = random.randint(0, self.width), random.randint(0, self.height)
            while abs(x - self.ship_x) < 100 and abs(y - self.ship_y) < 100:
                x, y = random.randint(0, self.width), random.randint(0, self.height)
            dx, dy = random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)
            size = random.randint(30, 90)
            self.asteroids.append([x, y, dx, dy, size])

        self.steps = 0
        self.done = False

    def reset(self, seed=None, options=None):
        self._init_game()
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # --- Apply action ---
        rotation_speed = 5
        acceleration = 0
        bullet_speed = 5

        if action == 1:  # rotate left
            self.ship_angle += rotation_speed
        elif action == 2:  # rotate right
            self.ship_angle -= rotation_speed
        elif action == 3:  # thrust
            acceleration = 0.2
        elif action == 4:  # shoot
            reward -= 1  # small penalty for shooting
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
                    reward += 100 - size  # smaller asteroids give more reward
                    if size > 30:
                        # split asteroid
                        for _ in range(2):
                            ndx, ndy = random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)
                            new_asteroids.append([ax, ay, ndx, ndy, size // 2])
                    break
            if not hit:
                new_asteroids.append(a)
        self.asteroids = new_asteroids

        # --- Collision ship vs asteroid ---
        ship_radius = self.ship_height / 2
        for a in self.asteroids:
            ax, ay, _, _, size = a
            if (ax - self.ship_x) ** 2 + (ay - self.ship_y) ** 2 < (size / 2 + ship_radius) ** 2:
                reward -= 100
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
