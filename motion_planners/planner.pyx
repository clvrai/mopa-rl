# distutils: language = c++
# distutils: sources = KinematicPlanner.cpp

from libcpp.string cimport string
from libcpp cimport bool as bool_
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "KinematicPlanner.h" namespace "MotionPlanner":
  cdef cppclass KinematicPlanner:
        KinematicPlanner(string, string, int, string, double, double, vector[int], vector[string],
                         vector[pair[int, int]], double, double, bool_, double, int) except +
        string xml_filename
        string opt
        int num_actions
        string algo
        double _range
        double threshold
        vector[int] passive_joint_idx
        vector[string] glue_bodies
        vector[pair[int, int]] ignored_contacts
        double contact_threshold
        string planner_status
        bool_ isSimplified
        double simplifiedDuration
        vector[vector[double]] plan(vector[double], vector[double], double)
        bool_ isValidState(vector[double])
        string getPlannerStatus()
        int seed

cdef class PyKinematicPlanner:
    cdef KinematicPlanner *thisptr
    def __cinit__(self, string xml_filename, string algo, int num_actions, string opt, double threshold,
                  double _range, vector[int] passive_joint_idx, vector[string] glue_bodies,
                  vector[pair[int, int]] ignored_contacts, double contact_threshold, double goal_bias,
                  bool_ is_simplified, double simplified_duration, int seed):

        self.thisptr = new KinematicPlanner(xml_filename, algo, num_actions, opt, threshold, _range,
                                           passive_joint_idx, glue_bodies, ignored_contacts, contact_threshold,
                                            goal_bias, is_simplified, simplified_duration, seed)

    def __dealloc__(self):
        del self.thisptr

    cpdef plan(self, start_vec, goal_vec, timelimit):
        return self.thisptr.plan(start_vec, goal_vec, timelimit)

    cpdef getPlannerStatus(self):
        return self.thisptr.getPlannerStatus()

    cpdef isValidState(self, state_vec):
        return self.thisptr.isValidState(state_vec)

