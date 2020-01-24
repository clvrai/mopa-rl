// Wrap MuJoCo functionality to make it available to OMPL API

#pragma once

#include <cmath>
#include <iostream>
#include <mutex>
#include <vector>

#include "mujoco.h"

struct JointInfo {
    std::string name;
    int type;
    bool limited;
    mjtNum range[2];
    int qposadr;
    int dofadr;
};

std::ostream& operator<<(std::ostream& os, const JointInfo& ji);

std::vector<JointInfo> getJointInfo(const mjModel* m);

struct StateRange {
    bool limited;
    mjtNum range[2];
};

StateRange getCtrlRange(const mjModel* m, size_t i);

struct MuJoCoState {
    mjtNum time;
    std::vector<mjtNum> qpos;
    std::vector<mjtNum> qvel;
    std::vector<mjtNum> act;
    std::vector<mjtNum> ctrl;
};

std::ostream& operator<<(std::ostream& os, const MuJoCoState& s);


// Put some sanity on the MuJoCo API
class MuJoCo {
  public:
    MuJoCo(std::string mjkey_filename):
      m(0), d(0) {
        // mj_activate and mj_deactivate should only be called
        // once per program
        // mj_instance_count_lock.lock();
        //if (mj_instance_count == 0) {
        mj_activate(mjkey_filename.c_str());
        //}
        //mj_instance_count += 1;
        //mj_instance_count_lock.unlock();
    }

    ~MuJoCo() {
        if (d) mj_deleteData(d);
        if (m) mj_deleteModel(m);
        mj_instance_count_lock.lock();
        mj_instance_count -= 1;
        if (mj_instance_count == 0) {
            mj_deactivate();
        }
        mj_instance_count_lock.unlock();
    }

    // TODO: copy constructor
    // TODO: assignment operator

    bool loadXML(std::string filename) {
        if (m) mj_deleteModel(m);
        char error[1000];
        m = mj_loadXML(filename.c_str(), 0, error, 1000);
        if (!m) {
            std::cerr << error << std::endl;
        }
        max_timestep = m->opt.timestep;
        return m;
    }

    bool makeData() {
        if (!m) {
            std::cerr << "Cannot makeData without a model!" << std::endl;
            return false;
        }
        if (d) mj_deleteData(d);
        d = mj_makeData(m);
        return d;
    }

    

    std::string getJointName(int i) const {
        // Avert your eyes of this horror
        return std::string(m->names + m->name_jntadr[i]);
    }

    std::string getBodyName(int i) const {
        return std::string(m->names + m->name_bodyadr[i]);
    }

    std::string getActName(int i) const {
        return std::string(m->names + m->name_actuatoradr[i]);
    }

    /// Set the world to random state within specified limits
    ///   modifies d->qpos and d->qvel
    void setRandomState() {
        mj_resetData(m, d);
        // Set default states
        for (size_t i=0; i < m->nq; i++) {
            d->qpos[i] = m->qpos0[i];
        }

        // Set random states within joint limit for DoFs
        auto joints = getJointInfo(m);
        for (size_t i=0; i < m->nv; i++) {
            int joint_id = m->dof_jntid[i];
            int qposadr = m->jnt_qposadr[ joint_id ];

            mjtNum r = ((mjtNum) rand()) / ((mjtNum) RAND_MAX);
            auto lower = joints[joint_id].range[0];
            auto upper = joints[joint_id].range[1];
            if (!joints[joint_id].limited) {
                // set range to -pi to pi
                lower = -3.1416;
                upper = 3.1416;
            }
            d->qpos[qposadr] = (r * (upper - lower)) + lower;

            // velocity = 0 seem reasonable
            d->qvel[i] = 0;
        }
    }

    void setState(MuJoCoState s) {
        d->time = s.time;
        for(size_t i=0; i < m->nq; i++) {
            if (i >= s.qpos.size()) break;
            d->qpos[i] = s.qpos[i];    
        }
        for(size_t i=0; i < m->nv; i++) {
            if (i >= s.qvel.size()) break;
            d->qvel[i] = s.qvel[i];    
        }
        for(size_t i=0; i < m->na; i++) {
            if (i >= s.act.size()) break;
            d->act[i] = s.act[i];    
        }
        for(size_t i=0; i < m->nu; i++) {
            if (i >= s.ctrl.size()) break;
            d->ctrl[i] = s.ctrl[i];    
        }
    }

    MuJoCoState getState() const {
        MuJoCoState s;
        s.time = d->time;
        for(size_t i=0; i < m->nq; i++) {
            s.qpos.push_back(d->qpos[i]);
        }
        for(size_t i=0; i < m->nv; i++) {
            s.qvel.push_back(d->qvel[i]);
        }
        for(size_t i=0; i < m->na; i++) {
            s.act.push_back(d->act[i]);
        }
        for(size_t i=0; i < m->nu; i++) {
            s.ctrl.push_back(d->ctrl[i]);
        }
        return s;
    }

    void step() {
        mj_step(m, d);
    }

    void sim_duration(double duration) {
        int steps = ceil(duration / max_timestep);
        m->opt.timestep = duration / steps;
        for(int i=0; i < steps; i++) {
            mj_step(m, d);
        }
    }

    double getMaxTimestep() const {
        return max_timestep;
    }

    mjModel* m;
    mjData* d;

  private:
    double max_timestep;
    static int mj_instance_count;
    static std::mutex mj_instance_count_lock;
};

