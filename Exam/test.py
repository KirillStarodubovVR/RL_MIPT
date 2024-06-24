import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((640, 480))

# Create a clock object
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with black
    screen.fill((0, 0, 0))

    # Draw a simple rectangle
    pygame.draw.rect(screen, (255, 0, 0), (300, 220, 40, 30))

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(60)  # The game will run at 60 FPS

# Quit Pygame
pygame.quit()
sys.exit()
