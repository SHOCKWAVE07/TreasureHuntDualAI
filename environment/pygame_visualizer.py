import pygame

def visualize_game(env, minimax_agent, dqn_agent):
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    clock = pygame.time.Clock()

    running = True
    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw grid, treasures, guards, and players
        for i in range(env.grid.shape[0]):
            for j in range(env.grid.shape[1]):
                rect = pygame.Rect(j * 50, i * 50, 50, 50)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
