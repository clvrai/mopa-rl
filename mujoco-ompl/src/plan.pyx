cimport c_planner
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef class Planner:
    def __cinit__(self, string xml_filename):
        self.thisptr = new c_planner.Planner(xml_filename)

    def __dealloc__(self):
        del self.thisptr

    cpdef plan(self, start_vec, goal_vec, timelimit):
        return self.thisptr.plan(start_vec, goal_vec, timelimit)
