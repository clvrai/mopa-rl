// Plan for a MuJoCo environment with OMPL

#include "Planner.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>


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
namespace og = ompl::geometric;

using namespace MotionPlanner;

Planner::Planner(std::string XML_filename)
{
    // std::string xml_filename = XML_filename;
    xml_filename = XML_filename;
}

Planner::~Planner(){
}
//
// void Planner::hello(){
//     cout << "hello";
// }
std::vector<std::vector<double> > Planner::planning(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit){
// double Planner::planning(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit){
    // Parse args with cxxargs
    std::string algo = "";
    double sst_selection_radius = -1.0;
    double sst_pruning_radius = -1.0;

    // Create MuJoCo Object
    std::string mjkey_filename = strcat(getenv("HOME"), "/.mujoco/mjkey.txt");
    auto mj(std::make_shared<MuJoCo>(mjkey_filename));

    // Get xml file name
    std::vector<std::vector<double>> failedSolutions(1, std::vector<double>(3, -1));
    if (xml_filename.find(".xml") == std::string::npos) {
        std::cerr << "XML model file is required" << std::endl;
        return failedSolutions;
        //return -1;
    }

    // Load Model
    std::cout << "Loading MuJoCo config from: " << xml_filename << std::endl;
    if (!mj->loadXML(xml_filename)) {
        std::cerr << "Could not load XML model file" << std::endl;
        // return solutions;
        //return -1;
        return failedSolutions;
    }

    // Make data
    if (!mj->makeData()) {
        std::cerr << "Could not allocate mjData" << std::endl;
        // return solutions;
        return failedSolutions;
        //return -1;
    }

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
        std::vector<ob::State*> &states =  p.getStates();
        int n = states.size();
        std::vector<std::vector<double> > solutions(n, std::vector<double>(start_vec.size(), -1));
        for (unsigned int i=0; i < n; ++i)
        {
            // const double *pos = states[i]->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0)->values;
            const ob::CompoundState* cState = states[i]->as<ob::CompoundState>();
            //solutions[i][0] = compoundState ->as<ob::RealVectorStateSpace::StateType>()->values[0];
            solutions[i][0] = cState -> as<ob::SO2StateSpace::StateType>(0)->value;

            for (unsigned int j=1; j < start_vec.size();  ++j){
                solutions[i][j] = cState -> as<ob::RealVectorStateSpace::StateType>(j)->values[0];
            }
            // solutions[i][1] = cState -> as<ob::RealVectorStateSpace::StateType>(1)->values[0];
        }



        // Write solution to file
        return solutions;
    }

    // return solutions;
    //return 0;
    return failedSolutions;
}
