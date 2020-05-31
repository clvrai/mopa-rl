# distutils: language = c++
# distutils: sources = KinematicPlanner.cpp

from libcpp.string cimport string
from libcpp cimport bool as bool_
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "KinematicPlanner.h" namespace "MotionPlanner":
  cdef cppclass KinematicPlanner:
        KinematicPlanner(string, string, int, double, double, string, double, double, double, vector[int], vector[string], vector[pair[int, int]], double, double, bool_, bool_, double) except +
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
        vector[int] passive_joint_idx
        vector[string] glue_bodies
        vector[pair[int, int]] ignored_contacts
        double contact_threshold
        string planner_status
        bool_ isSimplified
        double simplifiedDuration
        vector[vector[double]] plan(vector[double], vector[double], double, double, int)
        bool_ isValidState(vector[double])
        void removeCollision(int, int, int)
        string getPlannerStatus()
        bool_ allow_approximate

cdef class PyKinematicPlanner:
    cdef KinematicPlanner *thisptr
    def __cinit__(self, string xml_filename, string algo, int num_actions, double sst_selection_radius, double sst_pruning_radius, string opt, double threshold, double _range, double constructTime, vector[int] passive_joint_idx, vector[string] glue_bodies, vector[pair[int, int]] ignored_contacts, double contact_threshold, double goal_bias, bool_ allow_approximate, bool_ is_simplified, double simplified_duration):
        self.thisptr = new KinematicPlanner(xml_filename, algo, num_actions, sst_selection_radius, sst_pruning_radius, opt, threshold, _range, constructTime, passive_joint_idx, glue_bodies, ignored_contacts, contact_threshold, goal_bias, allow_approximate, is_simplified, simplified_duration)

    def __dealloc__(self):
        del self.thisptr

    cpdef plan(self, start_vec, goal_vec, timelimit, min_steps, attempts):
        return self.thisptr.plan(start_vec, goal_vec, timelimit, min_steps, attempts)

    cpdef removeCollision(self, geom_id, contype, conaffinity):
        return self.thisptr.removeCollision(geom_id, contype, conaffinity)

    cpdef getPlannerStatus(self):
        return self.thisptr.getPlannerStatus()

    cpdef isValidState(self, state_vec):
        return self.thisptr.isValidState(state_vec)

