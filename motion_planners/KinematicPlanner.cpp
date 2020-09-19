// Plan for a MuJoCo environment with OMPL

#include "KinematicPlanner.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>


#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/util/RandomNumbers.h>

#include <ompl/base/samplers/ObstacleBasedValidStateSampler.h>

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/objectives/StateCostIntegralObjective.h>
#include <ompl/base/objectives/MaximizeMinClearanceObjective.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <yaml-cpp/yaml.h>

#include <cxxopts.hpp>
#include "mujoco_wrapper.h"
#include "mujoco_ompl_interface.h"
#include "compound_state_projector.h"

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace og = ompl::geometric;

using namespace MotionPlanner;


KinematicPlanner::KinematicPlanner(std::string XML_filename, std::string Algo, int NUM_actions, std::string Opt,
                 double Threshold, double _Range, std::vector<int> Passive_joint_idx, std::vector<std::string> Glue_bodies,
                 std::vector<std::pair<int, int>> Ignored_contacts, double contact_threshold, double goal_bias,
                 bool is_simplified, double simplified_duration, int seed)
{
    // std::string xml_filename = XML_filename;
    ompl::msg::setLogLevel(ompl::msg::LOG_NONE); // OMPL logging
    // ompl::msg::setLogLevel(ompl::msg::LOG_DEBUG); // OMPL logging
    xml_filename = XML_filename;
    algo = Algo;
    num_actions = NUM_actions;
    opt = Opt;
    threshold = Threshold;
    passive_joint_idx = Passive_joint_idx;
    glue_bodies = Glue_bodies;
    ignored_contacts = Ignored_contacts;
    planner_status = "none";
    isSimplified = is_simplified;
    simplifiedDuration = simplified_duration;

    std::string homedir = std::getenv("HOME");
    mjkey_filename = homedir + "/.mujoco/mjkey.txt";
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
    }
// Make data
    if (!mj->makeData()) {
        std::cerr << "Could not allocate mjData" << std::endl;
    }

    // Setup OMPL environment
    si = MjOmpl::createSpaceInformationKinematic(mj->m, passive_joint_idx);

    msvc = std::make_shared<MjOmpl::MujocoStateValidityChecker>(si, mj, passive_joint_idx, false, ignored_contacts, contact_threshold);
    si->setStateValidityChecker(msvc);
    si->setStateValidityCheckingResolution(0.005);

    rrt_planner = std::make_shared<og::RRTstar>(si);
    rrt_connect_planner = std::make_shared<og::RRTConnect>(si);
    psimp_ = std::make_shared<og::PathSimplifier>(si);
    _range = _Range;


    ompl::RNG::setSeed(seed);
    si->setup();
    ss = std::make_shared<og::SimpleSetup>(si);

    if (algo == "rrt"){
        rrt_planner->setRange(_range);
        ss->setPlanner(rrt_planner);
    } else if (algo == "rrt_connect"){
        rrt_connect_planner->setRange(_range);
        ss->setPlanner(rrt_connect_planner);
    }

    if (opt == "maximize_min_clearance") {
        auto opt_obj(std::make_shared<ob::MaximizeMinClearanceObjective>(si));
        ss->setOptimizationObjective(opt_obj);
    } else if (opt == "path_length") {
        auto opt_obj(std::make_shared<ob::PathLengthOptimizationObjective>(si));
        ss->setOptimizationObjective(opt_obj);
    } else if (opt == "state_cost_integral") {
        auto opt_obj(std::make_shared<ob::StateCostIntegralObjective>(si));
        std::cout << "Cost: " << opt_obj->getCostThreshold().value() << std::endl;
        ss->setOptimizationObjective(opt_obj);
    }
    std::cout << "Finished planner setup" << std::endl;

}

KinematicPlanner::~KinematicPlanner(){
}

std::vector<std::vector<double> > KinematicPlanner::plan(std::vector<double> start_vec, std::vector<double> goal_vec,
                                                            double timelimit) {
    ss->clear();
    if (start_vec.size() != mj->m->nq) {
        std::cerr << "ERROR: start vector has dimension: " << start_vec.size()
        << " but should be nq: " << mj->m->nq;
    }
    if (goal_vec.size() != mj->m->nq) {
        std::cerr << "ERROR: goal vector has dimension: " << goal_vec.size()
             << " which should be nq: " << mj->m->nq;
    }
    if (algo == "rrt"){
        ss->getPlanner()->as<og::RRTstar>()->clear();
    } else if (algo == "rrt_connect"){
        ss->getPlanner()->as<og::RRTConnect>()->clear();
    }

    // split start_vec/goal_vec in active and passive dimensions
    std::vector<double> start_vec_active;
    std::vector<double> start_vec_passive;
    std::vector<double> goal_vec_active;
    std::vector<double> goal_vec_passive;
    for (int i=0; i<mj->m->nq; i++) {
        if (MjOmpl::isActiveJoint(i, passive_joint_idx)) {
            start_vec_active.push_back(start_vec[i]);
            goal_vec_active.push_back(goal_vec[i]);
        } else {
            start_vec_passive.push_back(start_vec[i]);
            goal_vec_passive.push_back(goal_vec[i]);
        }
    }

    ss->clearStartStates();
    auto initState = ss->getSpaceInformation()->allocState();
    MjOmpl::readOmplStateKinematic(start_vec_active,
                                   ss->getSpaceInformation().get(),
                                   initState->as<ob::CompoundState>());

    MjOmpl::copyOmplActiveStateToMujoco(initState->as<ob::CompoundState>(),
            ss->getSpaceInformation().get(), mj->m, mj->d, passive_joint_idx);

    MjOmpl::copyPassiveStateToMujoco(start_vec_passive, mj->m, mj->d, passive_joint_idx);

    msvc->addGlueTransformation(glue_bodies);

    // Set active start and goal states
    ob::ScopedState<> start_ss(ss->getStateSpace());
    for(int i=0; i < start_vec_active.size(); i++) {
        start_ss[i] = start_vec_active[i];
    }
    ob::ScopedState<> goal_ss(ss->getStateSpace());
    for(int i=0; i < goal_vec_active.size(); i++) {
        goal_ss[i] = goal_vec_active[i];
    }

    ss->setStartAndGoalStates(start_ss, goal_ss, threshold);
    if (!ss->getStateValidityChecker()->isValid(goal_ss.get())){
        std::vector<std::vector<double> > failedSolutions(1, std::vector<double>(start_vec.size(), -5));
        return failedSolutions;
    }
    // Call the planner
    ob::PlannerStatus solved;

    solved = ss->solve(timelimit);
    planner_status = solved.asString();

    if (bool(solved)){
        if (ss->haveExactSolutionPath()) {
            og::PathGeometric p = ss->getSolutionPath();
            if (isSimplified){
                psimp_->simplify(p, 5.0);
            }
            std::vector<ob::State*> &states =  p.getStates();
            int n = states.size();
            std::vector<std::vector<double>> solutions(n, std::vector<double>(start_vec.size(), -1));

            for (unsigned int i=0; i < n; ++i)
            {
                auto cState(states[i]->as<ob::CompoundState>());
                auto css(si->getStateSpace()->as<ob::CompoundStateSpace>());
                unsigned int n_passive = 0;
                unsigned int margin = 0;
                for (unsigned int j=0; j < start_vec.size();  ++j){
                    ss->getStateValidityChecker()->isValid(cState);
                    if (MjOmpl::isActiveJoint(j, passive_joint_idx)) {
                        unsigned int idx = j - n_passive - margin;
                        auto subspace(css->getSubspace(idx));
                        switch (subspace->getType()) {
                            case ob::STATE_SPACE_REAL_VECTOR:
                                solutions[i][j] = cState -> as<ob::RealVectorStateSpace::StateType>(idx)->values[0];
                                break;
                            case ob::STATE_SPACE_SO2:
                                solutions[i][j] = cState -> as<ob::SO2StateSpace::StateType>(idx)->value;
                                break;
                            default:
                                auto cSubState(cState -> as<ob::CompoundState>(idx));
                                margin += 6;
                                for (unsigned int k=0; k<3; ++k){
                                    solutions[i][j] = cSubState -> as<ob::RealVectorStateSpace::StateType>(0)->values[k];
                                    j++;
                                }
                                solutions[i][j] = cSubState -> as<ob::SO3StateSpace::StateType>(1)->x;
                                j++;
                                solutions[i][j] = cSubState -> as<ob::SO3StateSpace::StateType>(1)->y;
                                j++;
                                solutions[i][j] = cSubState -> as<ob::SO3StateSpace::StateType>(1)->z;
                                j++;
                                solutions[i][j] = cSubState -> as<ob::SO3StateSpace::StateType>(1)->w;
                                // j++;
                                break;
                        }
                    } else {
                        // set passive joints to their start value
                        solutions[i][j] = mj->d->qpos[j];
                        n_passive++;
                    }


                }
            }
            return solutions;
        }
    }

    std::vector<std::vector<double> > failedSolutions(1, std::vector<double>(start_vec.size(), -4));
    return failedSolutions;
}

bool KinematicPlanner::isValidState(std::vector<double> state_vec){
    if (algo == "rrt"){
        ss->getPlanner()->as<og::RRTstar>()->clear();
    } else if (algo == "rrt_connect"){
        ss->getPlanner()->as<og::RRTConnect>()->clear();
    }

    // split start_vec/goal_vec in active and passive dimensions
    std::vector<double> state_vec_active;
    std::vector<double> state_vec_passive;
    for (int i=0; i<mj->m->nq; i++) {
        if (MjOmpl::isActiveJoint(i, passive_joint_idx)) {
            state_vec_active.push_back(state_vec[i]);
        } else {
            state_vec_passive.push_back(state_vec[i]);
        }
    }

    ss->clearStartStates();
    auto initState = ss->getSpaceInformation()->allocState();
    MjOmpl::readOmplStateKinematic(state_vec_active,
                                   ss->getSpaceInformation().get(),
                                   initState->as<ob::CompoundState>());
    MjOmpl::copyOmplActiveStateToMujoco(initState->as<ob::CompoundState>(),
            ss->getSpaceInformation().get(), mj->m, mj->d, passive_joint_idx);
    MjOmpl::copyPassiveStateToMujoco(state_vec_passive, mj->m, mj->d, passive_joint_idx);
    msvc->addGlueTransformation(glue_bodies);
    // Set active start and goal states
    ob::ScopedState<> state_ss(ss->getStateSpace());
    for(int i=0; i < state_vec_active.size(); i++) {
        state_ss[i] = state_vec_active[i];
    }
    return ss->getStateValidityChecker()->isValid(state_ss.get());
}

std::string KinematicPlanner::getPlannerStatus(){
    // return ss->getLastPlannerStatus().asString();
    return planner_status;
}

