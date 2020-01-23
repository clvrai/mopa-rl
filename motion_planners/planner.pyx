# distutils: language = c++
# distutils: sources = Plan.cpp

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "Plan.h" namespace "MotionPlanner":
  cdef cppclass Planner:
        Planner(string, string, int, double, double, string) except +
        string xml_filename
        string opt
        int num_actions
        double sst_selection_radius
        double sst_pruning_radius
        string algo
        vector[vector[double]] planning(vector[double], vector[double], double)
        vector[vector[double]] planning_control(vector[double], vector[double], double)
        vector[vector[double]] kinematic_planning(vector[double], vector[double], double, double)

cdef class PyPlanner:
    cdef Planner *thisptr
    def __cinit__(self, string xml_filename, string algo, int num_actions, double sst_selection_radius, double sst_pruning_radius, string opt):
        self.thisptr = new Planner(xml_filename, algo, num_actions, sst_selection_radius, sst_pruning_radius, opt)

    def __dealloc__(self):
        del self.thisptr

    cpdef planning(self, start_vec, goal_vec, timelimit):
        return self.thisptr.planning(start_vec, goal_vec, timelimit)
    cpdef planning_control(self, start_vec, goal_vec, timelimit):
        return self.thisptr.planning_control(start_vec, goal_vec, timelimit)
    cpdef kinematic_planning(self, start_vec, goal_vec, timelimit, _range):
        return self.thisptr.kinematic_planning(start_vec, goal_vec, timelimit, _range)
