cimport c_planner
from c_planner cimport Planner as CPlanner
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef class Planner:
    cdef c_planner.Planner *thisptr
    cpdef plan(self, start_vec, goal_vec, timelimit)
