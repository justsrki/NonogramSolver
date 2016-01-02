from util.benchmark import timeit
from native.nonogram_solver_cy import *


class NonogramSolver:
    def __init__(self):
        self.values = None

    def set_values(self, values):
        self.values = values

    @timeit
    def get_result(self):
        col_cnt = len(self.values[0])
        row_cnt = len(self.values[1])

        try:
            native_solver = NonogramSolverNative()
            native_solver.set_values(row_cnt, col_cnt, self.values[1], self.values[0])
            native_solver.solve()
            sol = native_solver.get_solution()
        except Exception as e:
            print e
            sol = [[]]

        return sol
