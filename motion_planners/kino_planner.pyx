# distutils: language = c++
# distutils: sources = KinodynamicPlanner.cpp

from libcpp.string cimport string
from libcpp cimport bool as bool_
from libcpp.vector cimport vector

cdef extern from "KinodynamicPlanner.h" namespace "MotionPlanner":
  cdef cppclass KinodynamicPlanner:
        KinodynamicPlanner(string, string, int, double, double) except +
        string xml_filename
        int num_actions
        double sst_selection_radius
        double sst_pruning_radius

        vector[vector[double]] plan(vector[double], vector[double], double)

cdef class PyKinodynamicPlanner:
    cdef KinodynamicPlanner *thisptr
    def __cinit__(self, string xml_filename, string algo, int num_actions, double sst_selection_radius, double sst_pruning_radius):
        self.thisptr = new KinodynamicPlanner(xml_filename, algo, num_actions, sst_selection_radius, sst_pruning_radius)

    def __dealloc__(self):
        del self.thisptr

    cpdef plan(self, start_vec, goal_vec, timelimit):
        return self.thisptr.plan(start_vec, goal_vec, timelimit)
