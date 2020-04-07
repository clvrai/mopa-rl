import os, sys, mujoco_py
# workaround for mujoco py issue #390
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.
from util.contact_info import print_contact_info

# Choose your mujoco xml model
mj_path, _ = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
xml_path = '/home/gautam/HRLPlanner/hrl-planner/env/assets/xml/sawyer_pick_move_robosuite.xml'
# xml_path = '/home/gautam/HRLPlanner/hrl-planner/env/assets/xml/sawyer_test_robosuite.xml'

# Simulate
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
sim.step()
print(xml_path)

# Different debugging lines
i = 0
while True:
# for i in range(100):
    i = i+1
    sim.step()
    sim.render(mode='window')
    if i%10 == 0: # Log every 10 simulation steps
        print_contact_info(sim)
        # import ipdb; ipdb.set_trace()
        input("Press any key to re-render. Ctrl-C to quit")
