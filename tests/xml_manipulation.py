import os, sys
import numpy as np
import shutil

from motion_planners.sampling_based_planner import SamplingBasedPlanner
import env
import gym
from config import argparser
from env.inverse_kinematics import qpos_from_site_pose, qpos_from_site_pose_sampling
from config.motion_planner import add_arguments as planner_add_arguments
from math import pi
from util.misc import save_video
from util.gym import action_size
import time
import cv2
import xml.dom.minidom
import xml.etree.ElementTree as ET
import io


def render_frame(env, step, info={}):
    color = (200, 200, 200)
    text = "Step: {}".format(step)
    frame = env.render('rgb_array') * 255.0
    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

    if record_caption:
        font_size = 0.4
        thickness = 1
        offset = 12
        x, y = 5, fheight+10
        cv2.putText(frame, text,
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (255, 255, 0), thickness, cv2.LINE_AA)

        for i, k in enumerate(info.keys()):
            v = info[k]
            key_text = '{}: '.format(k)
            (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_size, thickness)
            cv2.putText(frame, key_text,
                        (x, y+offset*(i+2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (66, 133, 244), thickness, cv2.LINE_AA)
            cv2.putText(frame, str(v),
                        (x + key_width, y+offset*(i+2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


class MujocoXMLManipulator:
    def __init__(self, root):
        self.root = self.tree.getroot()
        self.name = self.root.get("model")
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.asset = self.create_default_element("asset")
        self.equality = self.create_default_element("equality")
        self.contact = self.create_default_element("contact")
        self.default = self.create_default_element("default")

        self.process_includes()

    def merge(self, other, merge_body=True):
        """
        Default merge method.
        Args:
            other: another MujocoXML instance
                raises XML error if @other is not a MujocoXML instance.
                merges <worldbody/>, <actuator/> and <asset/> of @other into @self
            merge_body: True if merging child bodies of @other. Defaults to True.
        """
        if not isinstance(other, MujocoXML):
            raise XMLError("{} is not a MujocoXML instance.".format(type(other)))
        if merge_body:
            for body in other.worldbody:
                self.worldbody.append(body)
        self.merge_asset(other)
        for one_actuator in other.actuator:
            self.actuator.append(one_actuator)
        for one_equality in other.equality:
            self.equality.append(one_equality)
        for one_contact in other.contact:
            self.contact.append(one_contact)
        for one_default in other.default:
            self.default.append(one_default)
        # self.config.append(other.config)

    def create_default_element(self, name):
        """
        Creates a <@name/> tag under root if there is none.
        """

        found = self.root.find(name)
        if found is not None:
            return found
        ele = ET.Element(name)
        self.root.append(ele)
        return ele


    def get_xml(self):
        """
        Returns a string of the MJCF XML file.
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()



parser = argparser()
args, unparsed = parser.parse_known_args()

if 'mover' in args.env:
    from config.mover import add_arguments
else:
    raise ValueError('args.env (%s) is not supported for this test script' % args.env)


add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

# Save video or not
is_save_video = True
record_caption = True

env = gym.make(args.env, **args.__dict__)
root = ET.fromstring(env.sim.model.get_xml())
body = root.find(".//body[@name='box']")
worldbody = root.find('worldbody')
worldbody.remove(body)
with io.StringIO() as string:
    string.write(ET.tostring(root, encoding='unicode'))
    print(string.getvalue())
import pdb
pdb.set_trace()
# XMLParser(env.xml_path)

