
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
        ob::RealVectorBounds bounds(1);
        bounds.setLow(joint.range[0]);
        bounds.setHigh(joint.range[1]);

        // Check that our assumptions are ok
        if (joint.qposadr != next_qpos) {
            cerr << "Uh oh......" << endl;
            throw invalid_argument(
                "Joints are not in order of qposadr ... write more code!");
        }
        next_qpos++;

        // Crate an appropriate subspace
        shared_ptr<ob::StateSpace> joint_space;
        switch(joint.type) {
          case mjJNT_FREE:
            joint_space = make_shared<ob::SE3StateSpace>();
            vel_spaces.push_back(make_shared<ob::RealVectorStateSpace>(6));
            next_qpos += 6;
            cerr << "Error: FREE joints are not yet supported!" << endl;
            throw invalid_argument(
                "FREE joints are not yet supported.");
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
    if (next_qpos != m->nq) {
        cerr << "ERROR: joint dims: " << next_qpos
             << " vs nq: " << m->nq << endl;
        throw invalid_argument("Total joint dimensions are not equal to nq");
    }

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
    auto space = makeCompoundStateSpace(m, true);

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
    const mjModel* m)
{

    auto space = makeCompoundStateSpace(m, false);

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
                    d->qpos[qpos_i] = substate
                        ->as<ob::RealVectorStateSpace::StateType>()->values[i];
                    qpos_i++;
                } else {
                    d->qvel[qvel_i] = substate
                        ->as<ob::RealVectorStateSpace::StateType>()->values[i];
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


void copySE3State(
        const double* data,
        ob::SE3StateSpace::StateType* state) {
    state->setX(data[0]);
    state->setY(data[1]);
    state->setZ(data[2]);
    copySO3State(data + 3, &state->rotation());
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

bool MujocoStateValidityChecker::isValid(const ompl::base::State *state) const {
    mj_lock.lock();
    if (si_->getStateSpace()->isCompound()) {
        copyOmplStateToMujoco(
            state->as<ob::CompoundState>(), si_, mj->m, mj->d, useVelocities);
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
    int ncon = mj->d->ncon;
    mj_lock.unlock();
    return ncon==0;
}

} // MjOmpl namespace
