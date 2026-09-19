class GridWorld:

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    def __init__(self, size=5, max_steps=100):
        self.size = size
        self.max_steps = max_steps

        self.agent = [0, 0]
        self.goal = [size - 1, size - 1]

        self.steps = 0

    def reset(self):
        self.agent = [0, 0]
        self.goal = [self.size - 1, self.size - 1]
        self.steps = 0

        return self.get_state()

    def get_state(self):
        return [
            self.agent[0] / (self.size - 1),
            self.agent[1] / (self.size - 1),
            self.goal[0] / (self.size - 1),
            self.goal[1] / (self.size - 1)
        ]

    def step(self, action):

        old_distance = self._distance_to_goal()

        if action == self.ACTION_UP:
            self.agent[1] += 1

        elif action == self.ACTION_DOWN:
            self.agent[1] -= 1

        elif action == self.ACTION_LEFT:
            self.agent[0] -= 1

        elif action == self.ACTION_RIGHT:
            self.agent[0] += 1

        # Limitar al mapa
        self.agent[0] = max(
            0,
            min(self.agent[0], self.size - 1)
        )

        self.agent[1] = max(
            0,
            min(self.agent[1], self.size - 1)
        )

        self.steps += 1

        new_distance = self._distance_to_goal()

        # Llegó al objetivo
        if self.agent == self.goal:
            reward = 10.0
            done = True

        # Se acabó el episodio
        elif self.steps >= self.max_steps:
            reward = -10.0
            done = True

        else:
            # Recompensa por acercarse
            if new_distance < old_distance:
                reward = -0.1
            else:
                reward = -0.3

            done = False

        return self.get_state(), reward, done

    def _distance_to_goal(self):
        dx = self.goal[0] - self.agent[0]
        dy = self.goal[1] - self.agent[1]

        return abs(dx) + abs(dy)

    def render(self):

        print()

        for y in reversed(range(self.size)):

            row = ""

            for x in range(self.size):

                if [x, y] == self.agent:
                    row += " A "

                elif [x, y] == self.goal:
                    row += " G "

                else:
                    row += " . "

            print(row)

        print()