#!/usr/bin/env python

import rospy

from geometry_msgs.msg import Twist, Quaternion #, Point, Pose, TwistWithCovariance, Vector3
import tf

import numpy as np

def main():
    rospy.init_node("sphero_simple_openloop")
    pub_twist = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    T = Twist()
    r = rospy.Rate(10)
    phi = 0
    # for i in range(1000):
    while not rospy.is_shutdown():
        T.linear.x = 50 * np.cos(phi/100.)
        T.linear.y = 50 * np.sin(2* phi/100.)
        pub_twist.publish(T)
        phi += 1
        r.sleep()
    print("a")


if __name__ == "__main__":
    main()    

    
