from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "Planner.h":
    cdef cppclass Planner:
        Planner(string)
        string xml_filename
        int plan(vector[double], vector[double], double)
