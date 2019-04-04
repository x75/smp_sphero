#!/usr/bin/env python
"""
quick joystick <-> sphero connector using ros to test basic connectivity
"""


import argparse
import rospy
import numpy as np
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class SpheroJoy(object):
    def __init__(self):
        self.sub = {}
        self.pub = {}
        self.sub["joy"] = rospy.Subscriber("/joy", Joy, self.cb_joy)
        self.sub["imu"] = rospy.Subscriber("/imu",
                                        Imu, self.cb_imu)
        self.pub["twist"] = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pub["color"] = rospy.Publisher("/set_color", ColorRGBA, queue_size=1)
        self.T = Twist()
        self.color = ColorRGBA()
        

    def cb_joy(self, msg):
        # print(msg)
        print(msg.axes[0])
        print(msg.axes[1])
        self.T.linear.x = msg.axes[1] * 255
        self.T.linear.y = msg.axes[0] * 255
        self.pub["twist"].publish(self.T)


    def cb_imu(self, msg):
        gain = 0.1
        clip = 255
        self.color.r = np.clip(msg.linear_acceleration.x * gain, 0, 1)
        self.color.g = np.clip(msg.linear_acceleration.y * gain, 0, 1)
        self.color.b = np.clip(msg.linear_acceleration.z * gain, 0, 1)
        # self.pub["color"].publish(self.color)

def main(args):
    rospy.init_node("sphero_joystick")
    r = rospy.Rate(20)
    sj = SpheroJoy()
    print("Main", args)
    while not rospy.is_shutdown():
        r.sleep()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
