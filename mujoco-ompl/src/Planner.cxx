// Plan for a MuJoCo environment with OMPL

#include "Planner.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include "boost/filesystem.hpp"

#include <ompl/control/SimpleSetup.h>
#include <ompl/control/planners/est/EST.h>
#include <ompl/control/planners/kpiece/KPIECE1.h>
#include <ompl/control/planners/pdst/PDST.h>
#include <ompl/control/planners/sst/SST.h>
#include <yaml-cpp/yaml.h>

#include <cxxopts.hpp>
#include "mujoco_wrapper.h"
#include "mujoco_ompl_interface.h"

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace filesystem = boost::filesystem;


Planner::Planner(std::string XML_filename)
{
    std::string xml_filename = XML_filename;
}

Planner::~Planner(){
}

int Planner::plan(vector<double> start_vec, vector<double> goal_vec, double timelimit){
    return 1;
}
