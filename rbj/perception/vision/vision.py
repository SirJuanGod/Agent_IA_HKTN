import random

import torch
from PIL import Image, ImageDraw


class VisionDatasetGenerator:

    def __init__(
        self,
        image_size=64,
        grid_size=5
    ):

        self.image_size = image_size
        self.grid_size = grid_size

    def generate(self):

        image = Image.new(
            "RGB",
            (
                self.image_size,
                self.image_size
            ),
            "black"
        )

        draw = ImageDraw.Draw(image)

        cell = self.image_size / self.grid_size

        # Posiciones aleatorias
        agent_x = random.randint(0, self.grid_size - 1)
        agent_y = random.randint(0, self.grid_size - 1)

        goal_x = random.randint(0, self.grid_size - 1)
        goal_y = random.randint(0, self.grid_size - 1)
        while goal_x == agent_x and goal_y == agent_y:
            goal_x = random.randint(0, self.grid_size - 1)
            goal_y = random.randint(0, self.grid_size - 1)

        obstacle_x = random.randint(0, self.grid_size - 1)
        obstacle_y = random.randint(0, self.grid_size - 1)

        # Evitar coincidencia con agente u objetivo
        while (
            (obstacle_x == agent_x and obstacle_y == agent_y) or
            (obstacle_x == goal_x and obstacle_y == goal_y)
        ):
            obstacle_x = random.randint(0, self.grid_size - 1)
            obstacle_y = random.randint(0, self.grid_size - 1)

        # Dibujar agente
        self._draw_cell(draw, agent_x, agent_y, cell, "white")

        # Dibujar objetivo
        self._draw_cell(draw, goal_x, goal_y, cell, "green")

        # Dibujar obstáculo
        self._draw_cell(draw, obstacle_x, obstacle_y, cell, "gray")

        # Label: estado estructurado normalizado (0 a 1)
        norm = max(1, self.grid_size - 1)
        label = (
            agent_x / norm,
            agent_y / norm,
            goal_x / norm,
            goal_y / norm,
            obstacle_x / norm,
            obstacle_y / norm
        )

        return image, label

    def _draw_cell(
        self,
        draw,
        x,
        y,
        cell,
        color
    ):

        x1 = int(x * cell)
        y1 = int(y * cell)

        x2 = int((x + 1) * cell)
        y2 = int((y + 1) * cell)

        draw.rectangle(
            [x1, y1, x2, y2],
            fill=color
        )