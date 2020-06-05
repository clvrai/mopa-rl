
#include <ompl/base/StateSpace.h>

#include "compound_state_projector.h"

#include "mujoco_ompl_interface.h"

namespace ob = ompl::base;
namespace oc = ompl::control;
using namespace std;

namespace MjOmpl {

void readOmplState(
        const vector<double>& x,
        const oc::SpaceInformation* si,
        ob::CompoundState* state,
        oc::RealVectorControlSpace::ControlType* control,
        double& duration) {
    // Vector format: [state, control, duration]
    // Interate over spaces, then controls, then duration

    assert(si->getStateSpace()->isCompound());
    auto css(si->getStateSpace()->as<ob::CompoundStateSpace>());
    auto rvcs(si->getControlSpace()->as<oc::RealVectorControlSpace>());

    // Make sure the data vector is the right size
    // cout << "x: " << x.size() << " css: " << css->getDimension()
    //      << " rvcs: " << rvcs->getDimension() << endl;
    assert(x.size() == css->getDimension() + rvcs->getDimension() + 1);

    int xpos = 0;

    // Read the state space
    for(size_t i=0; i < css->getSubspaceCount(); i++) {
        auto subspace(css->getSubspace(i));

        // Choose appropriate copy code based on subspace type
        size_t n;
        switch (subspace->getType()) {
          case ob::STATE_SPACE_REAL_VECTOR:
            n = subspace->as<ob::RealVectorStateSpace>()->getDimension();
            for(size_t j=0; j < n; j++) {
                (*state)[i]->as<ob::RealVectorStateSpace::StateType>()
                    ->values[j] = x[xpos];
                xpos++;
            }
            break;

          case ob::STATE_SPACE_SO2:
            (*state)[i]->as<ob::SO2StateSpace::StateType>()->value = x[xpos];
            xpos++;
            break;

          case ob::STATE_SPACE_SO3:
            copySO3State(
                x.data() + xpos,
                (*state)[i]->as<ob::SO3StateSpace::StateType>());
            xpos += 4;
            break;

          case ob::STATE_SPACE_SE3:
            copySE3State(
                x.data() + xpos,
                (*state)[i]->as<ob::SE3StateSpace::StateType>());
            xpos += 7;
            break;

          default:
            throw invalid_argument("Unhandled subspace type.");
            break;
        }
    }

    // Read the control space
    for(size_t i=0; i < rvcs->getDimension(); i++) {
        control->values[i] = x[xpos];
        xpos++;
    }

    // Read the duration
    duration = x[xpos];
    xpos++;

    assert(xpos == x.size());
}

void readOmplStateKinematic(
        const vector<double>& x,
        const ob::SpaceInformation* si,
        ob::CompoundState* state) {
    // Vector format: [state]

    assert(si->getStateSpace()->isCompound());
    auto css(si->getStateSpace()->as<ob::CompoundStateSpace>());

    // Make sure the data vector is the right size
    // cout << "x: " << x.size() << " css: " << css->getDimension()
    //      << " rvcs: " << rvcs->getDimension() << endl;
    assert(x.size() == css->getDimension());

    int xpos = 0;

    // Read the state space
    for(size_t i=0; i < css->getSubspaceCount(); i++) {
        auto subspace(css->getSubspace(i));

        // Choose appropriate copy code based on subspace type
        size_t n;
        switch (subspace->getType()) {
            case ob::STATE_SPACE_REAL_VECTOR:
                n = subspace->as<ob::RealVectorStateSpace>()->getDimension();
                for(size_t j=0; j < n; j++) {
                    (*state)[i]->as<ob::RealVectorStateSpace::StateType>()
                            ->values[j] = x[xpos];
                    xpos++;
                }
                break;

            case ob::STATE_SPACE_SO2:
                (*state)[i]->as<ob::SO2StateSpace::StateType>()->value = x[xpos];
                xpos++;
                break;

            case ob::STATE_SPACE_SO3:
                copySO3State(
                        x.data() + xpos,
                        (*state)[i]->as<ob::SO3StateSpace::StateType>());
                xpos += 4;
                break;

            case ob::STATE_SPACE_SE3:
                copySE3State(
                        x.data() + xpos,
                        (*state)[i]->as<ob::SE3StateSpace::StateType>());
                xpos += 7;
                break;

            default:
                throw invalid_argument("Unhandled subspace type.");
                break;
        }
    }

    assert(xpos == x.size());
}


shared_ptr<ob::CompoundStateSpace> makeCompoundStateSpace(
    const mjModel* m,
    const std::vector<int> &passive_joint_idx,
    bool include_velocity)
{
    //////////////////////////////////////////////
    // Create the state space (optionally including velocity)
    auto space(make_shared<ob::CompoundStateSpace>());

    // Iterate over joints
    auto joints = getJointInfo(m);
    // Add a subspace matching the topology of each joint
    vector<shared_ptr<ob::StateSpace> > vel_spaces;
    int next_qpos = 0;
    for(const auto& joint : joints) {
        // check if joint is a passive_joint
        bool ignore_joint = false;
        for (auto it = passive_joint_idx.begin(); it != passive_joint_idx.end(); ++it) {
            if (*it == joint.qposadr) {
                std::cout << "joint " << joint.name << " is ignored" << std::endl;
                ignore_joint=true;
                break;
            }
        }
        if (ignore_joint) {
            continue;
        }

        ob::RealVectorBounds bounds(1);
        bounds.setLow(joint.range[0]);
        bounds.setHigh(joint.range[1]);

        ob::RealVectorBounds se3bounds(3);
        // se3bounds.setLow(0, -0.00001);
        // se3bounds.setLow(1, -0.00001);
        // se3bounds.setLow(2, -0.00001);
        // se3bounds.setHigh(0, 0.00001);
        // se3bounds.setHigh(1, 0.00001);
        // se3bounds.setHigh(2, 0.00001);
        se3bounds.setLow(0, -10.);
        se3bounds.setLow(1, -10.);
        se3bounds.setLow(2, -10.);
        se3bounds.setHigh(0, 10.);
        se3bounds.setHigh(1, 10.);
        se3bounds.setHigh(2, 10.);

        // Check that our assumptions are ok
//        if (joint.qposadr != next_qpos) {
//            cerr << "Uh oh......" << endl;
//            throw invalid_argument(
//                "Joints are not in order of qposadr ... write more code!");
//        }
        next_qpos++;

        // Crate an appropriate subspace
        shared_ptr<ob::StateSpace> joint_space;
        switch(joint.type) {
          case mjJNT_FREE:
            joint_space = make_shared<ob::SE3StateSpace>();
            joint_space->as<ob::SE3StateSpace>()->setBounds(se3bounds);
            vel_spaces.push_back(make_shared<ob::RealVectorStateSpace>(6));
            next_qpos += 6;
            break;

          case mjJNT_BALL:
            // MuJoCo quaterions take 4d in pos space and 3d in vel space
            joint_space = make_shared<ob::SO3StateSpace>();
            if (joint.limited) {
                cerr << "ERROR: OMPL bounds on SO3 spaces are not implemented!"
                     << endl;
            }
            vel_spaces.push_back(make_shared<ob::RealVectorStateSpace>(3));
            next_qpos += 3;
            cerr << "Error: BALL joints are not yet supported!" << endl;
            throw invalid_argument(
                "BALL joints are not yet supported.");
            break;

          case mjJNT_HINGE:
            if (joint.limited) {
                // A hinge with limits is R^1
                joint_space = make_shared<ob::RealVectorStateSpace>(1);
                static_pointer_cast<ob::RealVectorStateSpace>(joint_space)
                    ->setBounds(bounds);
            } else {
                // A hinge with continuous rotation needs to be treated as
                // SO2 so OMPL knows that it rotates back to the original
                // position
                joint_space = make_shared<ob::SO2StateSpace>();
            }
            vel_spaces.push_back(make_shared<ob::RealVectorStateSpace>(1));
            break;

          case mjJNT_SLIDE:
            joint_space = make_shared<ob::RealVectorStateSpace>(1);
            if (joint.limited) {
                static_pointer_cast<ob::RealVectorStateSpace>(joint_space)
                    ->setBounds(bounds);
            }
            vel_spaces.push_back(make_shared<ob::RealVectorStateSpace>(1));
            break;

          default:
            cerr << "ERROR: Unknown joint type!" << endl;
            throw invalid_argument("Unknown joint type");
            break;
        }
        space->addSubspace(joint_space, 1.0);
    }
    if (next_qpos != m->nq - passive_joint_idx.size()) {
        cerr << "ERROR: joint dims: " << next_qpos
             << " vs nq - size(passive_joints): " << m->nq - passive_joint_idx.size() << endl;
        throw invalid_argument("Total joint dimensions are not equal to nq - size(passive_joints)");
    }

    std::cout << "Number of joints in mujoco  " << m->nq << std::endl;
    std::cout << "Number of active joints in planner  " << next_qpos << std::endl;
    std::cout << "Number of passive joints in planner " << passive_joint_idx.size() << std::endl;

    if (include_velocity) {
        // Add on all the velocity spaces
        for(const auto& s : vel_spaces) {
            // Apparently OMPL needs bounds
            // 350 m/s is just a bit supersonic
            // 50 m/s is pretty fast for most robot parts
            s->as<ob::RealVectorStateSpace>()->setBounds(-50, 50);
            space->addSubspace(s, 1.0);
        }
    }
    space->lock();  // We are done

    return space;
}

std::shared_ptr<ompl::base::RealVectorStateSpace>
makeRealVectorStateSpace(const mjModel *m, bool include_velocity) {
    //////////////////////////////////////////////
    auto joints = getJointInfo(m);

    uint dim;
    if (include_velocity) {
        dim = 2 * joints.size();
    } else {
        dim = joints.size();
    }

    auto space(make_shared<ob::RealVectorStateSpace>(dim));

    ob::RealVectorBounds bounds(dim);
    int i = 0;
    for(const auto& joint : joints) {
        bounds.setLow(i, joint.range[0]);
        bounds.setHigh(i, joint.range[1]);
        i++;
    }
    space->setBounds(bounds);
    return space;
}



shared_ptr<oc::SpaceInformation> createSpaceInformation(const mjModel* m) {
    std::vector<int> passive_joint_idx = {};  // passive joints are not implemented yet for kinodynamic planning
    auto space = makeCompoundStateSpace(m, passive_joint_idx, true);

    ////////////////////////////////
    // Create the control space
    int control_dim = m->nu;
    auto c_space(make_shared<oc::RealVectorControlSpace>(space, control_dim));
    // Set bounds
    ob::RealVectorBounds c_bounds(control_dim);
    c_bounds.setLow(-1);
    c_bounds.setHigh(1);
    // Handle specific bounds
    for(size_t i=0; i < control_dim; i++) {
        auto range = getCtrlRange(m, i);
        if (range.limited) {
            c_bounds.setLow(i, range.range[0]);
            c_bounds.setHigh(i, range.range[1]);
        }
    }
    c_space->setBounds(c_bounds);

    //////////////////////////////////////////
    // Set a default projection evaluator
    // auto proj_eval(
    //     make_shared<ompl::base::RealVectorRandomLinearProjectionEvaluator>(
    //         space, proj_dim));
    // auto proj_eval(make_shared<CompoundStateProjector>(space.get()));
    auto proj_eval = CompoundStateProjector::makeCompoundStateProjector(
        space.get());
    space->registerDefaultProjection(proj_eval);

    //////////////////////////////////////////
    // Combine into the SpaceInformation
    auto si(make_shared<oc::SpaceInformation>(space, c_space));
    si->setPropagationStepSize(m->opt.timestep);
    return si;
}


shared_ptr<ob::SpaceInformation> createSpaceInformationKinematic(
    const mjModel* m, const std::vector<int> &passive_joint_idx)
{

    auto space = makeCompoundStateSpace(m, passive_joint_idx, false);

    //////////////////////////////////////////
    // Set a default projection evaluator
    // TODO: this code leads to an error
    //    auto proj_eval = CompoundStateProjector::makeCompoundStateProjector(
    //        space.get());
    //    space->registerDefaultProjection(proj_eval);

    //////////////////////////////////////////
    // Combine into the SpaceInformation
    auto si(make_shared<ob::SpaceInformation>(space));
    return si;
}

void copyOmplActiveStateToMujoco(
        const ob::CompoundState* state,
        const ob::SpaceInformation* si,
        const mjModel* m,
        mjData* d,
        const std::vector<int> &passive_joint_idx) {
    // Iterate over subspaces to copy data from state to mjData
    // Copy position state to d->qpos
    assert(si->getStateSpace()->isCompound());
    auto css(si->getStateSpace()->as<ob::CompoundStateSpace>());

    // Iterate over joints
    int qpos_i = 0;
    for(size_t i=0; i < css->getSubspaceCount(); i++) {
        auto subspace(css->getSubspace(i));
        const ob::State* substate = (*state)[i];

        // Jump over passive joints
        while (!MjOmpl::isActiveJoint(qpos_i, passive_joint_idx)) {
            qpos_i++;
            if (qpos_i > m->nq) {
                throw invalid_argument("Passive joint indices do not match ompl subspace size");
            }
        }

        // Choose appropriate copy code based on subspace type
        size_t n;
        switch (subspace->getType()) {
            case ob::STATE_SPACE_REAL_VECTOR:
                n = subspace->as<ob::RealVectorStateSpace>()->getDimension();

                // Check if the vector does not align on the size of qpos
                // If this happens an assumption has been violated
                if (qpos_i < m->nq && (qpos_i + n) > m->nq) {
                    throw invalid_argument(
                            "RealVectorState does not align on qpos");
                }

                // Copy vector
                for(size_t i=0; i < n; i++) {
                    // Check if we copy to qpos or qvel
                    d->qpos[qpos_i] = substate->as<ob::RealVectorStateSpace::StateType>()->values[i];
                    qpos_i++;
                }
                break;

            case ob::STATE_SPACE_SO2:
                if (qpos_i >= m->nq) {
                    throw invalid_argument(
                            "SO2 velocity state should not happen.");
                }

                d->qpos[qpos_i] = substate->as<ob::SO2StateSpace::StateType>()->value;
                qpos_i++;
                break;

            case ob::STATE_SPACE_SO3:
                if (qpos_i + 4 > m->nq) {
                    throw invalid_argument("SO3 space overflows qpos");
                }

                copySO3State(substate->as<ob::SO3StateSpace::StateType>(), d->qpos + qpos_i);
                // That's right MuJoCo, ponter math is what you get when
                // you write an API like that :P
                qpos_i += 4;
                break;

            case ob::STATE_SPACE_SE3:
                if (qpos_i + 7 > m->nq) {
                    throw invalid_argument("SE3 space overflows qpos");
                }

                copySE3State(substate->as<ob::SE3StateSpace::StateType>(), d->qpos + qpos_i);
                qpos_i += 7;
                break;

            default:
                throw invalid_argument("Unhandled subspace type.");
                break;
        }
    }
}

void copyPassiveStateToMujoco(
        const std::vector<double>& passive_state,
        const mjModel* m,
        mjData* d,
        const std::vector<int> &passive_joint_idx) {
    auto joints = getJointInfo(m);
    int qpos_i = 0;
    for(const auto& joint : joints) {
        if (!isActiveJoint(joint.qposadr, passive_joint_idx)) {
          switch(joint.type) {
            case mjJNT_FREE:
              for (int i=0; i<7; i++) {
                d->qpos[joint.qposadr+i] = passive_state[qpos_i];
                qpos_i++;
              }
              break;

            case mjJNT_BALL:
              for (int i=0; i<4; i++) {
                d->qpos[joint.qposadr+i] = passive_state[qpos_i];
                qpos_i++;
              }
              break;

            case mjJNT_HINGE:
              d->qpos[joint.qposadr] = passive_state[qpos_i];
              qpos_i+=1;
              break;

            case mjJNT_SLIDE:
              d->qpos[joint.qposadr] = passive_state[qpos_i];
              qpos_i+=1;
              break;
          }
        }
    }
}

void copyOmplStateToMujoco(
        const ob::CompoundState* state,
        const ob::SpaceInformation* si,
        const mjModel* m,
        mjData* d,
        bool useVelocities) {
    // Iterate over subspaces to copy data from state to mjData
    // Copy position state to d->qpos
    // Copy velocity state to d->qvel
    assert(si->getStateSpace()->isCompound());
    auto css(si->getStateSpace()->as<ob::CompoundStateSpace>());

    int qpos_i = 0;
    int qvel_i = 0;
    for(size_t i=0; i < css->getSubspaceCount(); i++) {
        auto subspace(css->getSubspace(i));
        const ob::State* substate = (*state)[i];

        // Choose appropriate copy code based on subspace type
        size_t n;
        switch (subspace->getType()) {
          case ob::STATE_SPACE_REAL_VECTOR:
            n = subspace->as<ob::RealVectorStateSpace>()->getDimension();

            // Check if the vector does not align on the size of qpos
            // If this happens an assumption has been violated
            if (qpos_i < m->nq && (qpos_i + n) > m->nq) {
                throw invalid_argument(
                    "RealVectorState does not align on qpos");
            }

            // TODO: what is this checking for?
            if (!useVelocities && qpos_i >= m->nq) {
                throw invalid_argument(
                    "RealVectorState does not align on qpos"
                    " (useVelocities = false)");
            }

            // Copy vector
            for(size_t i=0; i < n; i++) {
                // Check if we copy to qpos or qvel
                if (qpos_i < m->nq) {
                    d->qpos[qpos_i] = substate->as<ob::RealVectorStateSpace::StateType>()->values[i];
                    qpos_i++;
                } else {
                    d->qvel[qvel_i] = substate->as<ob::RealVectorStateSpace::StateType>()->values[i];
                    qvel_i++;
                }
            }
            break;

          case ob::STATE_SPACE_SO2:
            if (qpos_i >= m->nq) {
                throw invalid_argument(
                    "SO2 velocity state should not happen.");
            }

            d->qpos[qpos_i]
                = substate->as<ob::SO2StateSpace::StateType>()->value;
            qpos_i++;
            break;

          case ob::STATE_SPACE_SO3:
            if (qpos_i + 4 > m->nq) {
                throw invalid_argument("SO3 space overflows qpos");
            }

            copySO3State(
                substate->as<ob::SO3StateSpace::StateType>(),
                d->qpos + qpos_i);
            // That's right MuJoCo, ponter math is what you get when
            // you write an API like that :P
            qpos_i += 4;
            break;

          case ob::STATE_SPACE_SE3:
            if (qpos_i + 7 > m->nq) {
                throw invalid_argument("SE3 space overflows qpos");
            }

            copySE3State(
                substate->as<ob::SE3StateSpace::StateType>(),
                d->qpos + qpos_i);
            qpos_i += 7;
            break;

          default:
            throw invalid_argument("Unhandled subspace type.");
            break;
        }
    }

    if (qpos_i != m->nq) {
        throw invalid_argument(
                "Size of data copied did not match m->nq");
    }

    if (useVelocities && (qvel_i != m->nv)) {
        throw invalid_argument(
                "Size of data copied did not match m->nv");
    }
}

void copyOmplStateToMujoco(
        const ob::RealVectorStateSpace::StateType* state,
        const ob::SpaceInformation* si,
        const mjModel* m,
        mjData* d,
        bool useVelocities) {
    if (useVelocities) {
        for (size_t i=0; i<si->getStateDimension(); i++) {
            if (i < si->getStateDimension() * 0.5 - 1) {
                d->qpos[i] = state->values[i];
            } else {
                d->qvel[i] = state->values[i];
            }
        }
    } else {
        for (size_t i=0; i<si->getStateDimension(); i++) {
            d->qpos[i] = state->values[i];
        }
    }
}

void copyMujocoStateToOmpl(
        const mjModel* m,
        const mjData* d,
        const ob::SpaceInformation* si,
        ob::CompoundState* state,
        bool useVelocities) {
    // Iterate over subspaces and copy data from mjData to CompoundState
    assert(si->getStateSpace()->isCompound());
    auto css(si->getStateSpace()->as<ob::CompoundStateSpace>());
    int qpos_i = 0;
    int qvel_i = 0;
    for(size_t i=0; i < css->getSubspaceCount(); i++) {
        auto subspace(css->getSubspace(i));

        // Choose appropriate copy code based on subspace type
        size_t n;
        switch (subspace->getType()) {
          case ob::STATE_SPACE_REAL_VECTOR:
            n = subspace->as<ob::RealVectorStateSpace>()->getDimension();

            // Check if the vector does not align on the size of qpos
            // If this happens an assumption has been violated
            if (qpos_i < m->nq && (qpos_i + n) > m->nq) {
                throw invalid_argument(
                    "RealVectorState does not align on qpos");
            }

            if (!useVelocities && qpos_i >= m->nq) {
                throw invalid_argument(
                    "RealVectorState does not align on qpos"
                    " (useVelocities = false)");
            }

            // Copy vector
            for(size_t j=0; j < n; j++) {
                // Check if we copy to qpos or qvel
                if (qpos_i < m->nq) {
                    (*state)[i]->as<ob::RealVectorStateSpace::StateType>()
                        ->values[j] = d->qpos[qpos_i];
                    qpos_i++;
                } else {
                    (*state)[i]->as<ob::RealVectorStateSpace::StateType>()
                        ->values[j] = d->qvel[qvel_i];
                    qvel_i++;
                }
            }
            break;

          case ob::STATE_SPACE_SO2:
            if (qpos_i >= m->nq) {
                throw invalid_argument(
                    "SO2 velocity state should not happen.");
            }

            (*state)[i]->as<ob::SO2StateSpace::StateType>()
                ->value = d->qpos[qpos_i];
            qpos_i++;
            break;

          case ob::STATE_SPACE_SO3:
            if (qpos_i + 4 > m->nq) {
                throw invalid_argument("SO3 space overflows qpos");
            }

            copySO3State(
                d->qpos + qpos_i,
                (*state)[i]->as<ob::SO3StateSpace::StateType>());
            qpos_i += 4;
            break;

          case ob::STATE_SPACE_SE3:
            if (qpos_i + 7 > m->nq) {
                throw invalid_argument("SE3 space overflows qpos");
            }

            copySE3State(
                d->qpos + qpos_i,
                (*state)[i]->as<ob::SE3StateSpace::StateType>());
            qpos_i += 7;
            break;

          default:
            throw invalid_argument("Unhandled subspace type.");
            break;
        }
    }

    if (qpos_i != m->nq) {
        throw invalid_argument(
                "Size of data copied did not match m->nq");
    }

    if (useVelocities && (qvel_i != m->nv)) {
        throw invalid_argument(
                "Size of data copied did not match m->nv");
    }
}


void copyOmplControlToMujoco(
        const oc::RealVectorControlSpace::ControlType* control,
        const oc::SpaceInformation* si,
        const mjModel* m,
        mjData* d) {
    int dim = si->getControlSpace()->as<oc::RealVectorControlSpace>()
        ->getDimension();
    if (dim != m->nu) {
        throw invalid_argument(
            "SpaceInformation and mjModel do not match in control dim");
    }

    for(size_t i=0; i < dim; i++) {
        d->ctrl[i] = control->values[i];
    }
}


void copySO3State(
        const ob::SO3StateSpace::StateType* state,
        double* data) {
    data[0] = state->w;
    data[1] = state->x;
    data[2] = state->y;
    data[3] = state->z;
}


void copySO3State(
        const double* data,
        ob::SO3StateSpace::StateType* state) {
    state->w = data[0];
    state->x = data[1];
    state->y = data[2];
    state->z = data[3];
}


void copySE3State(
        const ob::SE3StateSpace::StateType* state,
        double* data) {
    data[0] = state->getX();
    data[1] = state->getY();
    data[2] = state->getZ();
    copySO3State(&state->rotation(), data + 3);
}

pair<int, int> make_ordered_pair(int s1, int s2) {
    if (s1 < s2) {
        return pair<int, int>(s1, s2);
    }
    else {
        return pair<int, int>(s2, s1);
    }
}

void copySE3State(
        const double* data,
        ob::SE3StateSpace::StateType* state) {
    state->setX(data[0]);
    state->setY(data[1]);
    state->setZ(data[2]);
    copySO3State(data + 3, &state->rotation());
}

bool isActiveJoint(int idx, const std::vector<int> &passive_joint_idx) {
    for (auto it = passive_joint_idx.begin(); it != passive_joint_idx.end(); ++it) {
        if (*it == idx) {
            return false;
        }
    }
    return true;
}

void MujocoStatePropagator::propagate( const ob::State* state,
                                       const oc::Control* control,
                                       double duration,
                                       ob::State* result) const {
    //cout << " -- propagate asked for a timestep of: " << duration << endl;

    mj_lock.lock();
    copyOmplStateToMujoco(state->as<ob::CompoundState>(), si_, mj->m, mj->d);
    copyOmplControlToMujoco(
        control->as<oc::RealVectorControlSpace::ControlType>(),
        si_,
        mj->m,
        mj->d);

    mj->sim_duration(duration);

    // Copy result to ob::State*
    copyMujocoStateToOmpl(mj->m, mj->d, si_, result->as<ob::CompoundState>());

    mj_lock.unlock();
}


GlueTransformation::GlueTransformation(std::string body_a,
                                     std::string body_b,
                                     const mjModel *m,
                                     mjData *d) {
  mj_forward(m, d);

  id_body_a = mj_name2id(m, mjOBJ_BODY, body_a.c_str());
  id_body_b = mj_name2id(m, mjOBJ_BODY, body_b.c_str());

  if (id_body_a < 0) {
      std::cerr << "body_a id invalid " << body_a << std::endl;
  }
  if (id_body_b < 0) {
      std::cerr << "body_b id invalid " << body_b << std::endl;
  }

  jntNum = m->body_jntnum[id_body_b];
  jntAdr = m->body_jntadr[id_body_b];

  if (jntNum != 2 && jntNum != 1) {
      std::cerr << "invalid number of joints " << jntNum << std::endl;
  }
  if (jntNum < 0) {
      std::cerr << "invalid joint address " << jntAdr << std::endl;
  }

  // Calculate and store the current transformation between the two bodies
  mjtNum box_pos[3];
  mju_copy3(box_pos, &d->xpos[id_body_b*3]);

  mjtNum box_quat[4];
  mju_copy4(box_quat, &d->xquat[id_body_b*4]);

  mjtNum gripper_pos[3];
  mju_copy3(gripper_pos, &d->xpos[id_body_a*3]);
  mjtNum gripper_quat[4];
  mju_copy4(gripper_quat, &d->xquat[id_body_a*4]);
  mjtNum gripper_quat_inv[4];
  mju_negQuat(gripper_quat_inv , gripper_quat);

  mjtNum box_gripper_trans[3];
  mju_sub3(box_gripper_trans, box_pos, gripper_pos);
  mju_rotVecQuat(body_b_a_trans_g, box_gripper_trans, gripper_quat_inv);

  mju_mulQuat(body_b_a_rot, gripper_quat_inv, box_quat);
}

void GlueTransformation::applyTransformation(const mjModel *m, mjData *d) {
  if (jntNum == 2) {
      // two translational joints
      // get current body_a position + rotation
      mjtNum curr_body_a_pos[3];
      mju_copy3(curr_body_a_pos, &d->xpos[id_body_a*3]);
      mjtNum curr_body_a_rot[4];
      mju_copy4(curr_body_a_rot, &d->xquat[id_body_a*4]);

      mjtNum curr_body_a_b_trans[3];
      mju_rotVecQuat(curr_body_a_b_trans, body_b_a_trans_g, curr_body_a_rot);

      mjtNum new_body_b_pos[3];
      mju_add3(new_body_b_pos, curr_body_a_pos, curr_body_a_b_trans);
      d->qpos[jntAdr] = new_body_b_pos[0];
      d->qpos[jntAdr+1] = new_body_b_pos[1];

      // rotation
      //    mjtNum next_body_b_rot[4];
      //    mju_mulQuat(next_body_b_rot, curr_body_a_rot, body_b_a_rot);
      //     double dir = next_body_b_rot[3]/sqrt(next_body_b_rot[1]*next_body_b_rot[1] + next_body_b_rot[2]*next_body_b_rot[2] + next_body_b_rot[3]*next_body_b_rot[3]);
      //     mj->d->qpos[id_box_rot] = dir * (2.0*atan2(sqrt(next_body_b_rot[1]*next_body_b_rot[1] + next_body_b_rot[2]*next_body_b_rot[2] + next_body_b_rot[3]*next_body_b_rot[3]), next_body_b_rot[0]));
  } else if (jntNum == 1) {
      // free joint
      // get current body_a position + rotation
      mjtNum curr_body_a_pos[3];
      mju_copy3(curr_body_a_pos, &d->xpos[id_body_a*3]);
      mjtNum curr_body_a_rot[4];
      mju_copy4(curr_body_a_rot, &d->xquat[id_body_a*4]);

      mjtNum curr_body_a_b_trans[3];
      mju_rotVecQuat(curr_body_a_b_trans, body_b_a_trans_g, curr_body_a_rot);

      // set new position
      mjtNum new_body_b_pos[3];
      mju_add3(new_body_b_pos, curr_body_a_pos, curr_body_a_b_trans);
      d->qpos[jntAdr] = new_body_b_pos[0];
      d->qpos[jntAdr+1] = new_body_b_pos[1];
      d->qpos[jntAdr+2] = new_body_b_pos[2];

      // set new rotation
      mjtNum next_body_b_rot[4];
      mju_mulQuat(next_body_b_rot, curr_body_a_rot, body_b_a_rot);
      d->qpos[jntAdr+3] = next_body_b_rot[0];
      d->qpos[jntAdr+4] = next_body_b_rot[1];
      d->qpos[jntAdr+5] = next_body_b_rot[2];
      d->qpos[jntAdr+6] = next_body_b_rot[3];
  } else {
      std::cerr << "glue transformation not implemented yet for jntNum " <<  jntNum << std::endl;
  }
}

bool MujocoStateValidityChecker::isValid(const ompl::base::State *state) const {
    return isValid(state, this->ignored_contacts);
}

//Check if geoms outside of `ignored_contacts` are in collision. If yes, state is invalid. Otherwise state is valid.
// ignored_contacts MUST HAVE an ordered pair of ints
// This should even work for an empty vector of ignored contacts
// TODO: Can be made more performant
bool MujocoStateValidityChecker::isValid(const ompl::base::State *state, std::vector<pair<int, int>> ignored_contacts) const {
    mj_lock.lock();
    bool isValidState = true;  //valid until proven otherwise //return this after you unlock `mj_lock`
    if (si_->getStateSpace()->isCompound()) {
        copyOmplActiveStateToMujoco(
            state->as<ob::CompoundState>(), si_, mj->m, mj->d, passive_joint_idx);
    } else {
        copyOmplStateToMujoco(
            state->as<ob::WrapperStateSpace::StateType>()->getState()
                ->as<ob::RealVectorStateSpace::StateType>(),
            si_,
            mj->m,
            mj->d,
            useVelocities);
    }
    mj_fwdPosition(mj->m, mj->d);

    if (glue_transformation) {
        glue_transformation->applyTransformation(mj->m, mj->d);
        mj_fwdPosition(mj->m, mj->d);
    }

    int ncon = mj->d->ncon;
    if (ignored_contacts.size() == 0){
        for (int i = 0; i < ncon; i++) {
          mjContact con_data = mj->d->contact[i];
          if (con_data.dist <= contact_threshold) {
            isValidState = false;
//            break;
          }
        }
//        isValidState = ncon == 0;
    } else {
        for (int i = 0; i < ncon; i++) {
            mjContact con_data = mj->d->contact[i];
            pair<int, int> to_find = make_ordered_pair(con_data.geom1, con_data.geom2);

            bool found = false;
            for (auto it1 : ignored_contacts) {
                if (it1 == to_find) {
                    found = true;
                }
            }

            if (found == false) {
                if (con_data.dist <= contact_threshold) {
                    //current contact not found in list of ignored contacts
                    OMPL_DEBUG("Contact dist: %d\n", con_data.dist);
                    OMPL_DEBUG("Contact between geomIDs %d and %d unexpected. Invalid state\n", to_find.first, to_find.second);
                    isValidState = false; //so it's an invalid state
                }
            }
        }
    }

    if (isValidState) {
        //if all existing contacts are in the list of ignored contacts, state is valid
        OMPL_DEVMSG1("All contacts part of ignored contacts. %d ignored contacts. Valid state\n", ignored_contacts.size());
    }
    mj_lock.unlock();
    return isValidState;
}

} // MjOmpl namespace
