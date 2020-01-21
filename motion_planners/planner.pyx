# distutils: language = c++
# distutils: sources = Planner.cpp

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "Planner.h" namespace "MotionPlanner":
  cdef cppclass Planner:
        Planner(string) except +
        string xml_filename
        vector[vector[double]] planning(vector[double], vector[double], double)

cdef class PyPlanner:
    cdef Planner *thisptr
    def __cinit__(self, string xml_filename):
        self.thisptr = new Planner(xml_filename)

    def __dealloc__(self):
        del self.thisptr

    cpdef planning(self, start_vec, goal_vec, timelimit):
        return self.thisptr.planning(start_vec, goal_vec, timelimit)
