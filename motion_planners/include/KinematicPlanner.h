#ifndef _KinematicPlanner_h
#define _KinematicPlanner_h

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
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/pdst/PDST.h>
#include <ompl/geometric/planners/sst/SST.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/prm/SPARS.h>

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
    class KinematicPlanner
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
            std::shared_ptr<ob::SpaceInformation> si;
            std::shared_ptr<MjOmpl::MujocoStatePropagator> mj_state_prop;
            std::shared_ptr<og::RRTstar> rrt_planner;
            std::shared_ptr<og::SST> sst_planner;
            std::shared_ptr<og::PDST> pdst_planner;
            std::shared_ptr<og::EST> est_planner;
            std::shared_ptr<og::KPIECE1> kpiece_planner;
            std::shared_ptr<og::RRTConnect> rrt_connect_planner;
            std::shared_ptr<og::PRMstar> prm_star_planner;
            std::shared_ptr<og::SPARS> spars_planner;
            std::shared_ptr<og::SimpleSetup> ss;
            std::shared_ptr<MjOmpl::MujocoStateValidityChecker> msvc;
            double constructTime;
            bool is_construct;
            std::vector<int> passive_joint_idx;
            std::vector<std::string> glue_bodies;
            std::vector<std::pair<int,int>> ignored_contacts;

            KinematicPlanner(std::string xml_filename, std::string algo, int num_actions, double sst_selection_radius, double sst_pruning_radius, std::string opt, double threshold, double _range, double constructTime, std::vector<int> passive_joint_idx, std::vector<std::string> Glue_bodies, std::vector<std::pair<int, int>> ignored_contacts);
            ~KinematicPlanner();
            std::vector<std::vector<double> > plan(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit, double max_steps);
            void removeCollision(int geom_id, int contype, int conaffinity);
    };
}

#endif
