import pygame

class PygameVisualizer:
    def __init__(self, env):
        self.env = env
        self.cell_size = 50
        self.grid_size = env.grid.shape[0]
        self.screen_size = self.cell_size * self.grid_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Treasure Hunt Game")
        self.colors = {
            "empty": (50, 50, 50),
            "treasure": (255, 223, 0),
            "guard": (255, 0, 0),
            "minimax": (0, 255, 0),
            "dqn": (0, 0, 255),
        }

    def render(self):
        self.screen.fill((0, 0, 0))  # Clear the screen

        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw treasures
        for treasure in self.env.treasures:
            pygame.draw.circle(
                self.screen,
                self.colors["treasure"],
                (treasure[1] * self.cell_size + self.cell_size // 2, treasure[0] * self.cell_size + self.cell_size // 2),
                self.cell_size // 4,
            )

        # Draw guards
        for guard in self.env.guards:
            pygame.draw.rect(
                self.screen,
                self.colors["guard"],
                pygame.Rect(
                    guard[1] * self.cell_size + 10, guard[0] * self.cell_size + 10, self.cell_size - 20, self.cell_size - 20
                ),
            )

        # Draw players
        minimax_pos = self.env.minimax_player
        dqn_pos = self.env.dqn_player

        pygame.draw.rect(
            self.screen,
            self.colors["minimax"],
            pygame.Rect(
                minimax_pos[1] * self.cell_size + 10, minimax_pos[0] * self.cell_size + 10, self.cell_size - 20, self.cell_size - 20
            ),
        )

        pygame.draw.rect(
            self.screen,
            self.colors["dqn"],
            pygame.Rect(
                dqn_pos[1] * self.cell_size + 10, dqn_pos[0] * self.cell_size + 10, self.cell_size - 20, self.cell_size - 20
            ),
        )

        pygame.display.flip()
