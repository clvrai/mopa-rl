
#include "mujoco_wrapper.h"

using namespace std;


//////////////////////////////
// Init static variables
//////////////////////////////

int MuJoCo::mj_instance_count=0;
std::mutex MuJoCo::mj_instance_count_lock;

//////////////////////////////
// Define functions
//////////////////////////////

std::ostream& operator<<(std::ostream& os, const JointInfo& ji) {
    os << "Joint( name: \"" << ji.name << "\", "
       << "type: " << ji.type << ", "
       << "limited: " << ji.limited << ", "
       << "range: (" << ji.range[0] << ", " << ji.range[1] << ") "
       << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const MuJoCoState& s) {
    os << "{time: " << s.time << ", "
       << "qpos: [";
    for(auto const& i : s.qpos) {
        os << i << ", ";
    }
    os << "] qvel: [";
    for(auto const& i : s.qvel) {
        os << i << ", ";
    }
    os << "] act: [";
    for(auto const& i : s.act) {
        os << i << ", ";
    }
    os << "] ctrl: [";
    for(auto const& i : s.ctrl) {
        os << i << ", ";
    }
    os << "]}";
    return os;
}

std::vector<JointInfo> getJointInfo(const mjModel* m) {
    std::vector<JointInfo> joints;
    for (size_t i=0; i < m->njnt; i++) {
        JointInfo joint;
        joint.name = std::string(m->names + m->name_jntadr[i]);
        joint.type = m->jnt_type[i];
        joint.limited = (bool) m->jnt_limited[i];
        joint.range[0] = m->jnt_range[2*i];
        joint.range[1] = m->jnt_range[2*i + 1];
        joint.qposadr = m->jnt_qposadr[i];
        joint.dofadr = m->jnt_dofadr[i];
        joints.push_back(joint);
    }
    return joints;
}

StateRange getCtrlRange(const mjModel* m, size_t i) {
    StateRange r;
    r.limited = (bool) m->actuator_ctrllimited[i];
    r.range[0] = m->actuator_ctrlrange[2*i];
    r.range[1] = m->actuator_ctrlrange[2*i + 1];
    return r;
}
