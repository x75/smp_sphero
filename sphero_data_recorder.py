#!/usr/bin/env python
"""
Generate and execute random motor values in open-loop and record an episode
of sensorimotor data for use in offline experimentation and exploratory data anlysis

2015-2017 Oswald Berthold

"""

import rospy
import threading, signal, sys

from geometry_msgs.msg import Twist, Quaternion #, Point, Pose, TwistWithCovariance, Vector3
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
import tf

import numpy as np

class SpheroProbeEnvironment(threading.Thread):
    """Simple controller to probe the environment"""
    def __init__(self):
        threading.Thread.__init__(self)
        self.name = "SpheroProbeEnvironment"
        rospy.init_node(self.name)
        self.sub_imu = rospy.Subscriber("/imu",
                                        Imu, self.cb_imu)
        self.sub_odom = rospy.Subscriber("/odom",
                                        Odometry, self.cb_odom)
        self.pub_twist = rospy.Publisher("/cmd_vel", Twist, queue_size=2)
        self.T = Twist()
        self.r = rospy.Rate(20)
        self.isrunning = True
        self.runcnt = 0

        self.motor_target_ = np.zeros((2,1))
        self.motor_target = np.zeros((2,1))

        # record sensorimotor data
        self.experiment_len = 2000
        # imu.angular, imu.linear_acc, imu.orientation, odom.pose.pose.position,
        # odom.pose.pose.orientation, odom.twist.twist.angular, odom.twist.twist.linear, cmd_vel.Twist.linear
        self.numsmdata = 3 + 3 + 4 + 3 + 4 + 3 + 3 + 2
        self.smdata = np.zeros((self.experiment_len, self.numsmdata))

    def cb_imu(self, msg):
        # print msg
        # pass
        self.smdata[self.runcnt,0] = msg.angular_velocity.x
        self.smdata[self.runcnt,1] = msg.angular_velocity.y
        self.smdata[self.runcnt,2] = msg.angular_velocity.z
        self.smdata[self.runcnt,3] = msg.linear_acceleration.x
        self.smdata[self.runcnt,4] = msg.linear_acceleration.y
        self.smdata[self.runcnt,5] = msg.linear_acceleration.z
        self.smdata[self.runcnt,6] = msg.orientation.w
        self.smdata[self.runcnt,7] = msg.orientation.x
        self.smdata[self.runcnt,8] = msg.orientation.y
        self.smdata[self.runcnt,9] = msg.orientation.z

    def cb_odom(self, msg):
        # print msg
        # pass
        self.smdata[self.runcnt,10] = msg.pose.pose.position.x
        self.smdata[self.runcnt,11] = msg.pose.pose.position.y
        self.smdata[self.runcnt,12] = msg.pose.pose.position.z
        self.smdata[self.runcnt,13] = msg.pose.pose.orientation.w
        self.smdata[self.runcnt,14] = msg.pose.pose.orientation.x
        self.smdata[self.runcnt,15] = msg.pose.pose.orientation.y
        self.smdata[self.runcnt,16] = msg.pose.pose.orientation.z
        self.smdata[self.runcnt,17] = msg.twist.twist.angular.x
        self.smdata[self.runcnt,18] = msg.twist.twist.angular.y
        self.smdata[self.runcnt,19] = msg.twist.twist.angular.z
        self.smdata[self.runcnt,20] = msg.twist.twist.linear.x
        self.smdata[self.runcnt,21] = msg.twist.twist.linear.y
        self.smdata[self.runcnt,22] = msg.twist.twist.linear.z

    def run(self):
        while self.isrunning:
            print(("run cycle #%d" % (self.runcnt)))
            if self.runcnt % 20 == 0:
                self.motor_target_ = self.motor_target.copy()
                self.motor_target[0,0] = np.random.randint(20, 100)
                self.motor_target[1,0] = np.random.randint(20, 100)
                if np.random.uniform() > 0.5:
                    self.motor_target[0,0] *= -1
                if np.random.uniform() > 0.5:
                    self.motor_target[1,0] *= -1
                dx = (self.motor_target[0,0] - self.motor_target_[0,0]) / 20.
                dy = (self.motor_target[1,0] - self.motor_target_[1,0]) / 20.
            self.T.linear.x += dx
            self.T.linear.y += dy
            print(("self.T", self.T))
            self.smdata[self.runcnt,23] = self.T.linear.x
            self.smdata[self.runcnt,24] = self.T.linear.y
            
            self.pub_twist.publish(self.T)
            if self.runcnt == (self.experiment_len-1):
                np.save("smdata.npy", self.smdata)
                self.isrunning = False
                
            self.r.sleep()
                
            self.runcnt += 1
            
def main():
    s = SpheroProbeEnvironment()
    r = rospy.Rate(20)
    def handler(signum, frame):
        print('Signal handler called with signal', signum)
        s.isrunning = False
        # sys.exit(0)
        # raise IOError("Couldn't open device!")
    
    signal.signal(signal.SIGINT, handler)
    
    s.start()
    while not rospy.is_shutdown() and s.isrunning:
        # print("main loop")
        # time.sleep(1)
        r.sleep()
    
if __name__ == "__main__":
    main()    
