# distutils: language = c++
# distutils: sources = nonogram_solver.cpp
from libcpp.vector cimport vector


cdef extern from "nonogram_solver.hpp":
    cdef cppclass NonogramSolver:
        NonogramSolver() except +
        void setValues(int, int, vector[vector[int]], vector[vector[int]])
        void solve()
        vector[vector[int]] getSolution()


cdef class NonogramSolverNative:
    cdef NonogramSolver *this_ptr
    def __cinit__(self):
        self.this_ptr = new NonogramSolver()

    def __dealloc__(self):
        del self.this_ptr

    def set_values(self, row_cnt, col_cnt, row_vals, col_vals):
        self.this_ptr.setValues(row_cnt, col_cnt, row_vals, col_vals)

    def solve(self):
        self.this_ptr.solve()

    def get_solution(self):
        return self.this_ptr.getSolution()
