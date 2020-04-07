# distutils: language = c++
# distutils: sources = KinematicPlanner.cpp

from libcpp.string cimport string
from libcpp cimport bool as bool_
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "KinematicPlanner.h" namespace "MotionPlanner":
  cdef cppclass KinematicPlanner:
        KinematicPlanner(string, string, int, double, double, string, double, double, double, vector[pair[int, int]]) except +
        string xml_filename
        string opt
        int num_actions
        double sst_selection_radius
        double sst_pruning_radius
        string algo
        double _range
        double threshold
        double constructTime
        bool_ is_construct
        vector[pair[int, int]] ignored_contacts

        vector[vector[double]] plan(vector[double], vector[double], double, double)
        void removeCollision(int, int, int)

cdef class PyKinematicPlanner:
    cdef KinematicPlanner *thisptr
    def __cinit__(self, string xml_filename, string algo, int num_actions, double sst_selection_radius, double sst_pruning_radius, string opt, double threshold, double _range, double constructTime, vector[pair[int, int]] ignored_contacts):
        self.thisptr = new KinematicPlanner(xml_filename, algo, num_actions, sst_selection_radius, sst_pruning_radius, opt, threshold, _range, constructTime, ignored_contacts)

    def __dealloc__(self):
        del self.thisptr

    cpdef plan(self, start_vec, goal_vec, timelimit, max_steps):
        return self.thisptr.plan(start_vec, goal_vec, timelimit, max_steps)

    cpdef removeCollision(self, geom_id, contype, conaffinity):
        return self.thisptr.removeCollision(geom_id, contype, conaffinity)
