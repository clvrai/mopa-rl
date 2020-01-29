#ifndef _KinodynamicPlanner_h
#define _KinodynamicPlanner_h

#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include <ompl/control/SimpleSetup.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/control/planners/est/EST.h>
#include <ompl/control/planners/kpiece/KPIECE1.h>
#include <ompl/control/planners/pdst/PDST.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/pdst/PDST.h>
#include <ompl/geometric/planners/sst/SST.h>
#include <ompl/geometric/planners/prm/PRMstar.h>

#include <ompl/base/samplers/ObstacleBasedValidStateSampler.h>

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/objectives/StateCostIntegralObjective.h>
#include <ompl/base/objectives/MaximizeMinClearanceObjective.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <yaml-cpp/yaml.h>

#include <cxxopts.hpp>
#include "mujoco_wrapper.h"
#include "mujoco_ompl_interface.h"

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace og = ompl::geometric;

namespace MotionPlanner
{
    class KinodynamicPlanner
    {
        public:
            std::string xml_filename;
            std::string algo;
            int num_actions;
            double sst_selection_radius;
            double sst_pruning_radius;
            std::string opt;
            double threshold;
            double _range;
            std::string mjkey_filename;
            std::shared_ptr<MuJoCo> mj;

            KinodynamicPlanner(std::string xml_filename, std::string algo, int num_actions, double sst_selection_radius, double sst_pruning_radius);
            ~KinodynamicPlanner();
            std::vector<std::vector<double> > plan(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit);

    };
}

#endif
