/// Provide linkages between MuJoCo and OMPL

#pragma once

#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/WrapperStateSpace.h>

#include <ompl/control/StatePropagator.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <string>
#include <vector>
#include "mujoco_wrapper.h"

namespace MjOmpl {

/// Read a vectorized OMPL planning state back into data structures
/// Note: state and control must be pre-allocated from the space info
void readOmplState(
        const std::vector<double>& x,
        const ompl::control::SpaceInformation* si,
        ompl::base::CompoundState* state,
        ompl::control::RealVectorControlSpace::ControlType* control,
        double& duration);

void readOmplStateKinematic(
        const std::vector<double>& x,
        const ompl::base::SpaceInformation* si,
        ompl::base::CompoundState* state);


std::shared_ptr<ompl::control::SpaceInformation>
createSpaceInformation(const mjModel* m);

std::shared_ptr<ompl::base::SpaceInformation>
createSpaceInformationKinematic(const mjModel* m, const std::vector<int> &passive_joint_idx);

std::shared_ptr<ompl::base::CompoundStateSpace> makeCompoundStateSpace(
        const mjModel* m,
        const std::vector<int> &passive_joint_idx,
        bool include_velocity = true);

std::shared_ptr<ompl::base::RealVectorStateSpace> makeRealVectorStateSpace(
        const mjModel* m,
        bool include_velocity = true);

void copyOmplStateToMujoco(
        const ompl::base::RealVectorStateSpace::StateType*  state,
        const ompl::base::SpaceInformation* si,
        const mjModel* m,
        mjData* d,
        bool useVelocities=true);

void copyOmplStateToMujoco(
        const ompl::base::CompoundState* state,
        const ompl::base::SpaceInformation* si,
        const mjModel* m,
        mjData* d,
        bool useVelocities=true);

void copyOmplActiveStateToMujoco(
        const ompl::base::CompoundState* state,
        const ompl::base::SpaceInformation* si,
        const mjModel* m,
        mjData* d,
        const std::vector<int> &passive_joint_idx);

void copyPassiveStateToMujoco(
        const std::vector<double>& x,
        const mjModel* m,
        mjData* d,
        const std::vector<int> &passive_joint_idx);


void copyMujocoStateToOmpl(
        const mjModel* m,
        const mjData* d,
        const ompl::base::SpaceInformation* si,
        ompl::base::CompoundState* state,
        bool useVelocities=true);


void copyOmplControlToMujoco(
        const ompl::control::RealVectorControlSpace::ControlType* control,
        const ompl::control::SpaceInformation* si,
        const mjModel* m,
        mjData* d);

/// Copy SO3State to double array with no bounds checks
void copySO3State(
        const ompl::base::SO3StateSpace::StateType* state,
        double* data);

/// Copy double array to SO3 state with no bounds checks
void copySO3State(
        const double* data,
        ompl::base::SO3StateSpace::StateType* state);

/// Copy SE3State to double array with no bounds checks
void copySE3State(
        const ompl::base::SE3StateSpace::StateType* state,
        double* data);

/// Copy double array to SE3 state with no bounds checks
void copySE3State(
        const double* data,
        ompl::base::SE3StateSpace::StateType* state);

/// Checks if idx is an active or passive joint
bool isActiveJoint(
        int idx,
        const std::vector<int> &passive_joint_idx);

class MujocoStatePropagator : public ompl::control::StatePropagator {
  public:
    MujocoStatePropagator(
            std::shared_ptr<ompl::control::SpaceInformation> si,
            std::shared_ptr<MuJoCo> mj)
            : StatePropagator(si),
              mj(mj) {
    }

    const ompl::control::SpaceInformation* getSpaceInformation() const {
        return si_;
    }

    // To override this function from oc::StatePropagator, this has to be a
    // const function, but we need to modify the mjModel and mjData objects
    // to use MuJoCo to propagate a state
    // Use a preallocatd object, protect with mutex lock in case OMPL does
    // threading
    void propagate(
        const ompl::base::State* state,
        const ompl::control::Control* control,
        double duration,
        ompl::base::State* result) const override;

    bool canPropagateBackward() const override {
        return false;
    }

    bool canSteer() const override {
        return false;
    }

  private:
    // These have to be mutable because the tyrrany of OMPL makes
    // propagate a const function and I don't want to reallocate them
    mutable std::shared_ptr<MuJoCo> mj;
    mutable std::mutex mj_lock;
};

class GlueTransformation {
  public:
    // glue body_b to body_a
    GlueTransformation(std::string body_a,
                       std::string body_b,
                       const mjModel *m,
                       mjData *d);

    void applyTransformation(const mjModel *m, mjData *d);

    mjtNum body_b_a_trans_g[3];
    mjtNum body_b_a_rot[4];

    int jntNum;
    int jntAdr;
    int id_body_a;
    int id_body_b;
};

class MujocoStateValidityChecker : public ompl::base::StateValidityChecker {
  public:
    MujocoStateValidityChecker(
            const ompl::base::SpaceInformationPtr &si,
            std::shared_ptr<MuJoCo> mj,
            std::vector<int> passive_joint_idx,
            bool useVelocities=true,
            std::vector<std::pair<int, int>> ignored_contacts = {},
            double contact_threshold=0.0)
            : ompl::base::StateValidityChecker(si),
              mj(mj),
              passive_joint_idx(passive_joint_idx),
              ignored_contacts(ignored_contacts),
              useVelocities(useVelocities),
              contact_threshold(contact_threshold)
    {
    }

    void addGlueTransformation(const std::vector<std::string> &glue_bodies) {
        if (glue_bodies.size() > 0) {
            if (glue_transformation) {
              delete glue_transformation;
            }
            glue_transformation = new MjOmpl::GlueTransformation(glue_bodies[0], glue_bodies[1], mj->m, mj->d);
        }
    }

    bool isValid(const ompl::base::State *state) const;

  private:
    mutable std::shared_ptr<MuJoCo> mj;
    mutable std::mutex mj_lock;
    std::vector<int> passive_joint_idx;
    std::vector<std::pair<int, int>> ignored_contacts;
    bool useVelocities;
    GlueTransformation *glue_transformation = NULL;
    double contact_threshold;
    bool isValid(const ompl::base::State *state, std::vector<std::pair<int, int>> ignored_contacts) const;

};


} // MjOmpl namespace
