// Plan for a MuJoCo environment with OMPL

#include "KinematicPlanner.h"
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
#include <ompl/geometric/planners/prm/SPARS.h>

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


KinematicPlanner::KinematicPlanner(std::string XML_filename, std::string Algo, int NUM_actions, double SST_selection_radius, double SST_pruning_radius, std::string Opt,
                 double Threshold, double _Range, double constructTime, std::vector<int> Passive_joint_idx, std::vector<std::string> Glue_bodies, std::vector<std::pair<int, int>> Ignored_contacts, double contact_threshold)
{
    // std::string xml_filename = XML_filename;
    ompl::msg::setLogLevel(ompl::msg::LOG_NONE); // OMPL logging
    xml_filename = XML_filename;
    algo = Algo;
    sst_selection_radius = SST_selection_radius;
    sst_pruning_radius = SST_pruning_radius;
    num_actions = NUM_actions;
    opt = Opt;
    threshold = Threshold;
    constructTime = constructTime;
    is_construct = true;
    passive_joint_idx = Passive_joint_idx;
    glue_bodies = Glue_bodies;
    ignored_contacts = Ignored_contacts;

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
        // return solutions;
        //return -1;
    }

    // Make data
    if (!mj->makeData()) {
        std::cerr << "Could not allocate mjData" << std::endl;
        // return solutions;
        //return -1;
    }

    // Setup OMPL environment
    si = MjOmpl::createSpaceInformationKinematic(mj->m, passive_joint_idx);

    msvc = std::make_shared<MjOmpl::MujocoStateValidityChecker>(si, mj, passive_joint_idx, false, ignored_contacts, contact_threshold);
    si->setStateValidityChecker(msvc);

    rrt_planner = std::make_shared<og::RRTstar>(si);
    sst_planner = std::make_shared<og::SST>(si);
    pdst_planner = std::make_shared<og::PDST>(si);
    est_planner = std::make_shared<og::EST>(si);
    kpiece_planner = std::make_shared<og::KPIECE1>(si);
    rrt_connect_planner = std::make_shared<og::RRTConnect>(si);
    prm_star_planner = std::make_shared<og::PRMstar>(si);
    spars_planner = std::make_shared<og::SPARS>(si);
    _range = _Range;

    si->setup();
    ss = std::make_shared<og::SimpleSetup>(si);

    if (algo == "sst") {
        sst_planner->setSelectionRadius(sst_selection_radius); // default 0.2
        sst_planner->setPruningRadius(sst_pruning_radius); // default 0.1
        sst_planner->setRange(_range);
        ss->setPlanner(sst_planner);

        std::cout << "Using SST planner with selection radius ["
             << sst_selection_radius
             << "] and pruning radius ["
             << sst_pruning_radius
             << "]" << std::endl;
    } else if (algo == "pdst") {
        ss->setPlanner(pdst_planner);
    } else if (algo == "est") {
        est_planner->setRange(_range);
        ss->setPlanner(est_planner);
        est_planner->setup();
    } else if (algo == "kpiece") {
        kpiece_planner->setRange(_range);
        ss->setPlanner(kpiece_planner);
    } else if (algo == "rrt"){
        rrt_planner->setRange(_range);
        ss->setPlanner(rrt_planner);
    } else if (algo == "rrt_connect"){
        rrt_connect_planner->setRange(_range);
        ss->setPlanner(rrt_connect_planner);
    } else if (algo == "prm_star"){
        ss->setPlanner(prm_star_planner);
        ss->setup();
        ss->getPlanner()->as<og::PRMstar>()->constructRoadmap(ob::timedPlannerTerminationCondition(constructTime));
        std::cout << "Milestone: " << ss->getPlanner()->as<og::PRMstar>()->milestoneCount() << std::endl;
    } else if (algo == "spars"){
        ss->setPlanner(spars_planner);
        ss->setup();
        ss->getPlanner()->as<og::SPARS>()->constructRoadmap(ob::timedPlannerTerminationCondition(constructTime));
        std::cout << "Milestone: " << ss->getPlanner()->as<og::SPARS>()->milestoneCount() << std::endl;
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
                                                            double timelimit, double min_steps) {

    if (start_vec.size() != mj->m->nq) {
        std::cerr << "ERROR: start vector has dimension: " << start_vec.size()
        << " but should be nq: " << mj->m->nq;
    }
    if (goal_vec.size() != mj->m->nq) {
        std::cerr << "ERROR: goal vector has dimension: " << goal_vec.size()
             << " which should be nq: " << mj->m->nq;
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
            // std::cout << "joint i " << i << " active " << start_vec[i] << std::endl;
        } else {
            start_vec_passive.push_back(start_vec[i]);
            goal_vec_passive.push_back(goal_vec[i]);
            // std::cout << "joint i " << i << " passive " << start_vec[i] << std::endl;
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
        std::vector<std::vector<double> > failedSolutions(1, std::vector<double>(start_vec.size(), -1));
        OMPL_DEBUG("Invalid Goal");
        std::cout << "Invalid Goal" << std::endl;
        return failedSolutions;
    }

    // Call the planner
    ob::PlannerStatus solved;
    // if(is_clear){
    //     ss->clear();
    // }

    if (algo == "prm_star" || algo == "spars"){
        std::cout << "Milestone: " << ss->getPlanner()->as<og::PRMstar>()->milestoneCount() << std::endl;
    }
    solved = ss->solve(timelimit);

    // std::cout << "solved " << solved << std::endl;

    if (solved) {
        // ss.getSolutionPath().print(std::cout);
        // if (is_simplified){
        //     ss->simplifySolution(simplified_duration);
        // }
        og::PathGeometric p = ss->getSolutionPath();
        // ss->getSolutionPath().print(std::cout);
        // p.interpolate(min_steps);
        std::vector<ob::State*> &states =  p.getStates();
        int n = states.size();
        std::vector<std::vector<double>> solutions(n, std::vector<double>(start_vec.size(), -1));

        for (unsigned int i=0; i < n; ++i)
        {
            auto cState(states[i]->as<ob::CompoundState>());
            // solutions[i][0] = cState -> as<ob::SO2StateSpace::StateType>(0)->value;
            auto css(si->getStateSpace()->as<ob::CompoundStateSpace>());
            unsigned int n_passive = 0;
            // for (unsigned int j=0; j < css->getSubspaceCount();  ++j){
            //     ss->getStateValidityChecker()->isValid(cState);
            //     if (MjOmpl::isActiveJoint(j, passive_joint_idx)) {
            //     }
            // }
            unsigned int margin = 0;
            for (unsigned int j=0; j < start_vec.size();  ++j){
                ss->getStateValidityChecker()->isValid(cState);
                if (MjOmpl::isActiveJoint(j, passive_joint_idx)) {
                    unsigned int idx = j - n_passive - margin;
                    // std::cout << "start: " << start_vec.size() << " j: " << j << " idx: " << idx << std::endl;
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
                                // std::cout << "index: " << j << std::endl;
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
        // Write solution to file
        //ss->clear();
        if (algo == "sst") {
            ss->getPlanner()->as<og::SST>()->clear();
        } else if (algo == "pdst") {
            ss->getPlanner()->as<og::PDST>()->clear();
        } else if (algo == "est") {
            ss->getPlanner()->as<og::EST>()->clear();
        } else if (algo == "kpiece") {
            ss->getPlanner()->as<og::KPIECE1>()->clear();
        } else if (algo == "rrt"){
            ss->getPlanner()->as<og::RRTstar>()->clear();
        } else if (algo == "sst"){
            ss->getPlanner()->as<og::SST>()->clear();
        } else if (algo == "rrt_connect"){
            ss->getPlanner()->as<og::RRTConnect>()->clear();
        } else if (algo == "prm_star"){
            ss->getPlanner()->as<og::PRMstar>()->clearQuery();
        } else if (algo == "spars"){
            ss->getPlanner()->as<og::SPARS>()->clearQuery();
        }
        return solutions;
    }

    if (algo == "sst") {
        ss->getPlanner()->as<og::SST>()->clear();
    } else if (algo == "pdst") {
        ss->getPlanner()->as<og::PDST>()->clear();
    } else if (algo == "est") {
        ss->getPlanner()->as<og::EST>()->clear();
    } else if (algo == "kpiece") {
        ss->getPlanner()->as<og::KPIECE1>()->clear();
    } else if (algo == "rrt"){
        ss->getPlanner()->as<og::RRTstar>()->clear();
    } else if (algo == "sst"){
        ss->getPlanner()->as<og::SST>()->clear();
    } else if (algo == "rrt_connect"){
        ss->getPlanner()->as<og::RRTConnect>()->clear();
    } else if (algo == "prm_star"){
        ss->getPlanner()->as<og::PRMstar>()->clearQuery();
    } else if (algo == "spars"){
            ss->getPlanner()->as<og::SPARS>()->clearQuery();
    }

    // return solutions;
    //return 0;
    std::vector<std::vector<double> > failedSolutions(1, std::vector<double>(start_vec.size(), -1));
    return failedSolutions;
}


void KinematicPlanner::removeCollision(int geom_id, int contype, int conaffinity){
    mj->m->geom_contype[geom_id] = contype;
    mj->m->geom_conaffinity[geom_id] = conaffinity;
    mj_step(mj->m, mj->d);
    // Setup OMPL environment
    si = MjOmpl::createSpaceInformationKinematic(mj->m, passive_joint_idx);

    auto msvc = std::make_shared<MjOmpl::MujocoStateValidityChecker>(si, mj, passive_joint_idx, false);
    msvc->addGlueTransformation(glue_bodies);
    si->setStateValidityChecker(msvc);

    rrt_planner = std::make_shared<og::RRTstar>(si);
    sst_planner = std::make_shared<og::SST>(si);
    pdst_planner = std::make_shared<og::PDST>(si);
    est_planner = std::make_shared<og::EST>(si);
    kpiece_planner = std::make_shared<og::KPIECE1>(si);
    rrt_connect_planner = std::make_shared<og::RRTConnect>(si);
    prm_star_planner = std::make_shared<og::PRMstar>(si);
    spars_planner = std::make_shared<og::SPARS>(si);

    si->setup();
    ss = std::make_shared<og::SimpleSetup>(si);

    if (algo == "sst") {
        sst_planner->setSelectionRadius(sst_selection_radius); // default 0.2
        sst_planner->setPruningRadius(sst_pruning_radius); // default 0.1
        sst_planner->setRange(_range);
        ss->setPlanner(sst_planner);

        std::cout << "Using SST planner with selection radius ["
             << sst_selection_radius
             << "] and pruning radius ["
             << sst_pruning_radius
             << "]" << std::endl;
    } else if (algo == "pdst") {
        ss->setPlanner(pdst_planner);
    } else if (algo == "est") {
        est_planner->setRange(_range);
        ss->setPlanner(est_planner);
        est_planner->setup();
    } else if (algo == "kpiece") {
        kpiece_planner->setRange(_range);
        ss->setPlanner(kpiece_planner);
    } else if (algo == "rrt"){
        rrt_planner->setRange(_range);
        ss->setPlanner(rrt_planner);
    } else if (algo == "rrt_connect"){
        rrt_connect_planner->setRange(_range);
        ss->setPlanner(rrt_connect_planner);
    } else if (algo == "prm_star"){
        ss->setPlanner(prm_star_planner);
        ss->setup();
        ss->getPlanner()->as<og::PRMstar>()->constructRoadmap(ob::timedPlannerTerminationCondition(constructTime));
        std::cout << "Milestone: " << ss->getPlanner()->as<og::PRMstar>()->milestoneCount() << std::endl;
    } else if (algo == "spars"){
        ss->setPlanner(spars_planner);
        ss->setup();
        ss->getPlanner()->as<og::SPARS>()->constructRoadmap(ob::timedPlannerTerminationCondition(constructTime));
        std::cout << "Milestone: " << ss->getPlanner()->as<og::SPARS>()->milestoneCount() << std::endl;
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
}
