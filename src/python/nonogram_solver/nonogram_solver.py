import numpy as np


class NonogramSolver:
    def __init__(self):
        self.values = None

    def set_values(self, values):
        self.values = values

    def get_result(self):
        v = len(self.values[0])
        h = len(self.values[1])
        sol = np.random.randint(0, 2, size=(h, v))
        return sol
