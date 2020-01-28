// Plan for a MuJoCo environment with OMPL

#include "KinodynamicPlanner.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
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


using namespace MotionPlanner;


KinodynamicPlanner::KinodynamicPlanner(std::string XML_filename, std::string Algo, int NUM_actions, double SST_selection_radius, double SST_pruning_radius)
{
    // std::string xml_filename = XML_filename;
    xml_filename = XML_filename;
    algo = Algo;
    sst_selection_radius = SST_selection_radius;
    sst_pruning_radius = SST_pruning_radius;
    num_actions = NUM_actions;

    mjkey_filename = strcat(getenv("HOME"), "/.mujoco/mjkey.txt");
    mj = std::make_shared<MuJoCo>(mjkey_filename);

    // Get xml file name
    if (xml_filename.find(".xml") == std::string::npos) {
        std::cerr << "XML model file is required" << std::endl;
        //return -1;
    }

    // Load Model
    std::cout << "Loading MuJoCo config from: " << xml_filename << std::endl;
    if (!mj->loadXML(xml_filename)) {
        std::cerr << "Could not load XML model file" << std::endl;
        // return solutions;
        //return -1;
    }

    // Make data
    if (!mj->makeData()) {
        std::cerr << "Could not allocate mjData" << std::endl;
        // return solutions;
        //return -1;
    }
}

KinodynamicPlanner::~KinodynamicPlanner(){
}

std::vector<std::vector<double> > KinodynamicPlanner::plan(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit){
// double Planner::planning(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit){
    // Parse args with cxxargs
    // Setup OMPL environment
    auto si = MjOmpl::createSpaceInformation(mj->m);
    auto mj_state_prop(std::make_shared<MjOmpl::MujocoStatePropagator>(si, mj));
    si->setStatePropagator(mj_state_prop);

    // Create a SimpleSetup object
    oc::SimpleSetup ss(si);

    // Create some candidate planners
    auto sst_planner(std::make_shared<oc::SST>(si));
    auto pdst_planner(std::make_shared<oc::PDST>(si));
    auto est_planner(std::make_shared<oc::EST>(si));
    auto kpiece_planner(std::make_shared<oc::KPIECE1>(si));
    auto rrt_planner(std::make_shared<oc::RRT>(si));

    // TODO: change the optimization objective?
    // auto opt_obj(make_shared<ob::OptimizationObjective>(si));
    // ss.setOptimizationObjective(opt_obj);

    if (algo == "sst") {
        sst_planner->setSelectionRadius(sst_selection_radius); // default 0.2
        sst_planner->setPruningRadius(sst_pruning_radius); // default 0.1
        ss.setPlanner(sst_planner);

        std::cout << "Using SST planner with selection radius ["
             << sst_selection_radius
             << "] and pruning radius ["
             << sst_pruning_radius
             << "]" << std::endl;
    } else if (algo == "pdst") {
        ss.setPlanner(pdst_planner);
    } else if (algo == "est") {
        ss.setPlanner(est_planner);
        est_planner->setup();
    } else if (algo == "kpiece") {
        ss.setPlanner(kpiece_planner);
    } else if (algo == "rrt") {
        ss.setPlanner(rrt_planner);
    }

    // Set start and goal states
    ob::ScopedState<> start_ss(ss.getStateSpace());
    for(int i=0; i < start_vec.size(); i++) {
        start_ss[i] = start_vec[i];
    }


    ob::ScopedState<> goal_ss(ss.getStateSpace());
    for(int i=0; i < goal_vec.size(); i++) {
        goal_ss[i] = goal_vec[i];
    }
    double threshold = 0.1;
    ss.setStartAndGoalStates(start_ss, goal_ss, threshold);

    // Call the planner
    ob::PlannerStatus solved = ss.solve(timelimit);

    if (solved) {
        std::cout << "Found Solution with status: " << solved.asString() << std::endl;
        ss.getSolutionPath().print(std::cout);
        og::PathGeometric p = ss.getSolutionPath().asGeometric();
        //p.interpolate();
        std::vector<ob::State*> &states =  p.getStates();
        oc::PathControl pathControl = ss.getSolutionPath();
        //std::vector<oc::Control*> &controls = ss.getSolutionPath().getControls();
        int n = states.size();
        int control_n = pathControl.getControls().size();
        std::vector<std::vector<double> > solutions(n, std::vector<double>(start_vec.size(), -1));
        std::vector<std::vector<double> > solutionControls(control_n, std::vector<double>(num_actions, -1));

        for (unsigned int i=0; i<control_n; ++i){
            const double* u = pathControl.getControl(i)->as<oc::RealVectorControlSpace::ControlType>()->values;
            for (unsigned int j=0; j<num_actions; ++j){
                solutionControls[i][j] = u[j];
            }
        }
        return solutionControls;
    }

    // return solutions;
    //return 0;
    std::vector<std::vector<double> > failedSolutions(1, std::vector<double>(1, -1));
    return failedSolutions;
}
