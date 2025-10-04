import pygame
from asteroids_env.env import AsteroidsEnv

env = AsteroidsEnv(render_mode="human", width=400, height=400, max_steps=2000)
obs, _ = env.reset()

done = False
clock = pygame.time.Clock()

while not done:
    # --- Pump Pygame events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # --- Read key states ---
    pygame.event.pump()  # ensures key.get_pressed() updates
    keys = pygame.key.get_pressed()
    action = 0  # default: do nothing
    if keys[pygame.K_LEFT]:
        action = 1
    elif keys[pygame.K_RIGHT]:
        action = 2
    elif keys[pygame.K_UP]:
        action = 3
    elif keys[pygame.K_SPACE]:
        action = 4

    obs, reward, done, truncated, info = env.step(action)
    clock.tick(60)  # limit FPS to 60

env.close()
