#!/usr/bin/env python
"""**hk2.py**

Homeostasis and homeokinesis behaviours (compare Der and Martius 2012, Playful
machines) done with python via ROS for
 - Sphero (real robot)
 - lpzrobots barrel (simulated)

Run like
python hk.py --help

FIXME: put the learner / control structure into class to easily load
       der/martius or reservoir model

control: velocity and angle
control: raw motors
convenience: smp_thread_ros
convenience: ros based setters for parameters
convenience: publish preprocessed quantities
"""

import time, argparse, sys
import numpy as np
import scipy.sparse as spa
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32, Float32MultiArray, ColorRGBA
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion #, Point, Pose, TwistWithCovariance, Vector3

import tf

from smp_base.smp_thread import smp_thread_ros
from smp_base.models_reservoirs import Reservoir

################################################################################
# helper funcs
def dtanh(x):
    return 1 - np.tanh(x)**2

def idtanh(x):
    return 1./dtanh(x) # hm?    

def get_cb_dict(func):
    return {"func": func, "cnt": 0}
    
################################################################################
# robot classes
class robot(object):
    def __init__(self, ref):
        self.ref = ref
        self.cb   = {}
        self.pubs = {}
        self.subs = {
            "/sensors_exp": [Float64MultiArray,]
            }
        self.lag = 1
        
    def get_pubsub(self):
        """return pubs and subs dicts as tuple"""
        return (self.pubs, self.subs)
    
class robotSphero(robot):
    def __init__(self, ref):
        robot.__init__(self, ref)
        self.pubs = {
            "/cmd_vel": [Twist,],
            "/cmd_vel_raw": [Twist,],
            "/set_color": [ColorRGBA,],
            "/lpzros/x": [Float32MultiArray]
            }
        self.cb = {
            "/imu": get_cb_dict(self.cb_imu),
            "/odom": get_cb_dict(self.cb_odom),
            }
        
        self.subs = {
            "/imu": [Imu, self.cb["/imu"]["func"]],
            "/odom": [Odometry, self.cb["/odom"]["func"]],
            }
        # custom
        self.numsen_raw = 10 # 8 # 5 # 2
        self.numsen     = 10 # 8 # 5 # 2
        
        self.imu  = Imu()
        self.odom = Odometry()
        # sphero color
        self.color = ColorRGBA()
        self.motors = Twist()
        
        self.msg_inputs     = Float32MultiArray()
        self.msg_motors     = Float64MultiArray()
        self.msg_sensor_exp = Float64MultiArray()

        self.imu_lin_acc_gain = 0 # 1e-1
        self.imu_gyrosco_gain = 1e-1
        self.imu_orienta_gain = 0 # 1e-1
        self.linear_gain      = 0 # 1.0 # 1e-1
        self.pos_gain         = 0 # 1e-2
        self.output_gain = 255 # 120 # 120
        
        # sphero lag is 4 timesteps
        self.lag = 1 # 2
        
    def cb_imu(self, msg):
        """ROS IMU callback: use odometry and incoming imu data to trigger
        sensorimotor loop execution"""
        # print "imu", msg
        # FIXME: do the averaging here
        self.imu = msg
        imu_vec_acc = np.array((self.imu.linear_acceleration.x, self.imu.linear_acceleration.y, self.imu.linear_acceleration.z))
        imu_vec_gyr = np.array((self.imu.angular_velocity.x, self.imu.angular_velocity.y, self.imu.angular_velocity.z))
        (r, p, y) = tf.transformations.euler_from_quaternion([self.imu.orientation.x, self.imu.orientation.y, self.imu.orientation.z, self.imu.orientation.w])
        imu_vec_ori = np.array((r, p, y))
        imu_vec_ = np.hstack((imu_vec_acc, imu_vec_gyr, imu_vec_ori)).reshape(self.imu_vec.shape)
        self.imu_vec = self.imu_vec * self.imu_smooth + (1 - self.imu_smooth) * imu_vec_
        # print "self.imu_vec", self.imu_vec

        self.cb["/imu"]["cnt"] += 1
        
    def cb_odom(self, msg):
        """ROS odometry callback, copy incoming data into local memory"""
        # print type(msg)
        self.odom = msg        
        self.cb["odom"]["cnt"] += 1

    def prepare_inputs(self):
        inputs = (self.odom.twist.twist.linear.x * self.linear_gain, self.odom.twist.twist.linear.y * self.linear_gain,
                         self.imu_vec[0] * self.imu_lin_acc_gain,
                         self.imu_vec[1] * self.imu_lin_acc_gain,
                         self.imu_vec[2] * self.imu_lin_acc_gain,
                         self.imu_vec[3] * self.imu_gyrosco_gain,
                         self.imu_vec[4] * self.imu_gyrosco_gain,
                         self.imu_vec[5] * self.imu_gyrosco_gain,
                         self.odom.pose.pose.position.x * self.pos_gain,
                         self.odom.pose.pose.position.y * self.pos_gain,
                         )
        # print "%s.prepare_inputs inputs = %s" % (self.__class__.__name__, inputs)
        return np.array(inputs)

    def prepare_output(self, y):
        # self.motors.linear.x = y[0,0] * self.output_gain
        # self.motors.linear.y = y[1,0] * self.output_gain
        # self.pub["_cmd_vel"].publish(self.motors)
        self.motors.linear.x  = y[1,0] * self.output_gain * 1.414
        self.motors.angular.z = y[0,0] * 1 # self.output_gain
        self.pubs["_cmd_vel_raw"].publish(self.motors)
        print "%s.prepare_output y = %s , motors = %s" % (y, self.motors)
        
class robotLPZ(robot):
    def __init__(self, ref):
        robot.__init__(self, ref)
        self.pubs = {
            "/motors": [Float64MultiArray,],
            "/lpzros/x": [Float32MultiArray]
            }
        self.subs = {
            "/sensors": [Float64MultiArray, self.cb_sensors],
        }
        self.numsen_raw = 2
        self.numsen     = 2
        self.nummot     = 2
        self.sensors = Float64MultiArray()
        self.sensors.data = [0 for i in range(self.numsen_raw)]
        self.motors  = Float64MultiArray()
        self.motors.data = [0 for i in range(self.nummot)]
        # self.lag = 1
        
    def cb_sensors(self, msg):
        self.sensors = msg

    def prepare_inputs(self):
        inputs = np.array(self.sensors.data)
        print "%s.prepare_inputs inputs = %s" % (self.__class__.__name__, inputs)
        return inputs

    def prepare_output(self, y):
        self.motors.data = y
        # print "self.pubs", self.pubs
        self.ref.pub["_motors"].publish(self.motors)
        

################################################################################
# main homeostasis, homeokinesis class based on smp_thread_ros
class LPZRos(smp_thread_ros):
    modes = {"hs": 0, "hk": 1, "eh_pi_d": 2}

    def __init__(self, mode="hs", loop_time = 1./20, robot = "lpz"):
        print "loop_time", loop_time
        if robot == "lpz":
            self.robot = robotLPZ(self)
        elif robot == "sphero":
            self.robot = robotSphero(self)
        else:
            self.robot = None

        # get pubsub configuration
        pubs, subs = self.robot.get_pubsub()
        
        # init ros runners
        smp_thread_ros.__init__(self, loop_time = loop_time,
                                pubs = pubs, subs = subs)
        
        # self.name = "lpzros"
        self.mode = LPZRos.modes[mode]
        self.cnt = 0
    
        ############################################################
        # model + meta params
        self.numsen_raw = self.robot.numsen_raw # 10 # 8 # 5 # 2
        self.numsen = self.robot.numsen # 10 # 8 # 5 # 2
        self.nummot = 2
        # buffer size accomodates causal minimum 1 + lag time steps
        self.bufsize = 1 + self.robot.lag # 2
        self.creativity = 5.2
        self.epsA = 0.2
        # self.epsA = 0.02
        # self.epsA = 0.001
        # self.epsC = 0.001
        # self.epsC = 0.001
        # self.epsC = 0.01
        # self.epsC = 0.1
        self.epsC = 0.5
        # self.epsC = 0.9
        # self.epsC = 1.0
        # self.epsC = 2.0

        ############################################################
        # forward model
        # self.A = np.eye(self.numsen) * 1.
        self.A  = np.zeros((self.numsen, self.nummot))
        self.A[range(self.nummot),range(self.nummot)] = 1.
        self.b = np.zeros((self.numsen,1))
        # controller
        # self.C  = np.eye(self.nummot) * 0.4
        self.C  = np.zeros((self.nummot, self.numsen))
        self.C[range(self.nummot),range(self.nummot)] = 1 * 0.4
        # self.C  = np.random.uniform(-1e-2, 1e-2, (self.nummot, self.numsen))
        print "self.C", self.C
        self.h  = np.zeros((self.nummot,1))
        self.g  = np.tanh # sigmoidal activation function
        self.g_ = dtanh # derivative of sigmoidal activation function
        # state
        self.x = np.ones ((self.numsen, self.bufsize))
        self.y = np.zeros((self.nummot, self.bufsize))
        self.z = np.zeros((self.numsen, 1))
        # auxiliary variables
        self.L     = np.zeros((self.numsen, self.nummot))
        self.v_avg = np.zeros((self.numsen, 1)) 
        self.xsi   = np.zeros((self.numsen, 1))

        self.imu_vec  = np.zeros((3 + 3 + 3, 1))
        self.imu_smooth = 0.8 # coef
        
        # expansion
        self.exp_size = self.numsen
        self.exp_hidden_size = 100
        self.res = Reservoir(N = self.exp_hidden_size, p = 0.1, g = 1.5, tau = 0.1, input_num = self.numsen_raw, input_scale = 5.0)
        self.res_wo_expand     = np.random.randint(0, self.exp_hidden_size, self.exp_size)
        self.res_wo_expand_amp = np.random.uniform(0, 1, (self.exp_size, 1)) * 0.8
        self.res_wi_expand_amp = np.random.uniform(0, 1, (self.exp_size, self.numsen_raw)) * 1.0
        
    def expansion_random_system(self, x, dim_target = 1):
        # dim_source = x.shape[0]
        # print "x", x.shape
        self.res.execute(x)
        # print "self.res.r", self.res.r.shape
        a = self.res.r[self.res_wo_expand]
        # print "a.shape", a.shape
        b = a * self.res_wo_expand_amp
        # print "b.shape", b.shape
        c = b + np.dot(self.res_wi_expand_amp, x)
        return c
        
    def brain(self, msg):
        """lpz sensors callback: receive sensor values, sos algorithm attached, FloatArray input msg"""
        # FIXME: fix the timing
        # print "msg", msg
        now = 0
        # self.msg_motors.data = []
        self.x = np.roll(self.x, 1, axis=1) # push back past
        self.y = np.roll(self.y, 1, axis=1) # push back past
        # update with new sensor data
        self.x[:,now] = msg # np.array(msg)
        # self.msg_inputs.data = self.x[:,now].flatten().tolist()
        # self.pubs["_lpzros_x"].publish(self.msg_inputs)
        
        # self.x[[0,1],now] = 0.
        # print "msg", msg
        
        # xa = np.array([msg.data]).T
        # self.x[:,[0]] = self.expansion_random_system(xa, dim_target = self.numsen)
        # self.msg_sensor_exp.data = self.x.flatten().tolist()
        # self.pub_sensor_exp.publish(self.msg_sensor_exp)
        
        # compute new motor values
        x_tmp = np.atleast_2d(self.x[:,now]).T + self.v_avg * self.creativity
        # print "x_tmp.shape", x_tmp.shape
        # print self.g(np.dot(self.C, x_tmp) + self.h)
        m1 = np.dot(self.C, x_tmp)
        # print "m1.shape", m1.shape
        t1 = self.g(m1 + self.h).reshape((self.nummot,))
        self.y[:,now] = t1

        self.cnt += 1
        if self.cnt <= 2: return

        # print "x", self.x
        # print "y", self.y
        
        # local variables
        x = np.atleast_2d(self.x[:,self.robot.lag]).T
        # this is wrong
        # y = np.atleast_2d(self.y[:,self.lag]).T
        # this is better
        y = np.atleast_2d(self.y[:,self.robot.lag]).T
        x_fut = np.atleast_2d(self.x[:,now]).T

        # print "x", x.shape, x, x_fut.shape, x_fut
        z = np.dot(self.C, x + self.v_avg * self.creativity) + self.h
        # z = np.dot(self.C, x)
        # print z.shape, x.shape
        # print z - x

        g_prime = dtanh(z) # derivative of g
        g_prime_inv = idtanh(z) # inverse derivative of g

        # print "g_prime", self.cnt, g_prime
        # print "g_prime_inv", self.cnt, g_prime_inv

        # forward prediction error xsi
        # FIXME: include state x in forward model
        xsi = x_fut - (np.dot(self.A, y) + self.b)
        print "xsi =", xsi
        
        # forward model learning
        dA = self.epsA * np.dot(xsi, y.T) + (self.A * -0.0003) # * 0.1
        self.A += dA
        db = self.epsA * xsi              + (self.b * -0.0001) # * 0.1
        self.b += db

        # print "A", self.cnt, self.A
        # print "b", self.b

        if self.mode == 1: # TLE / homekinesis
            eta = np.dot(np.linalg.pinv(self.A), xsi)
            zeta = np.clip(eta * g_prime_inv, -1., 1.)
            # print "eta", self.cnt, eta
            # print "zeta", self.cnt, zeta
            # print "C C^T", np.dot(self.C, self.C.T)
            # mue = np.dot(np.linalg.pinv(np.dot(self.C, self.C.T)), zeta)
            # changed params + noise shape
            lambda_ = np.eye(self.nummot) * np.random.uniform(-0.01, 0.01, (self.nummot, self.nummot))
            mue = np.dot(np.linalg.pinv(np.dot(self.C, self.C.T) + lambda_), zeta)
            v = np.clip(np.dot(self.C.T, mue), -1., 1.)
            self.v_avg += (v - self.v_avg) * 0.1
            # print "v", self.cnt, v
            # print "v_avg", self.cnt, self.v_avg
            EE = 1.0

            # print EE, v
            if True: # logarithmic error
                # EE = .1 / (np.sqrt(np.linalg.norm(v)) + 0.001)
                EE = .1 / (np.square(np.linalg.norm(v)) + 0.001)
            # print EE
            # print "eta", eta
            # print "zeta", zeta
            # print "mue", mue
            
            dC = (np.dot(mue, v.T) + (np.dot((mue * y * zeta), -2 * x.T))) * EE * self.epsC
            dh = mue * y * zeta * -2 * EE * self.epsC

            # pass
            # dC = np.zeros_like(self.C)
            # dh = np.zeros_like(self.h)
            
        elif self.mode == 0: # homestastic learning
            eta = np.dot(self.A.T, xsi)
            print "eta", self.cnt, eta.shape, eta
            dC = np.dot(eta * g_prime, x.T) * self.epsC
            dh = eta * g_prime * self.epsC
            # print dC, dh
            # self.C +=

        # FIXME: ???
        self.h += np.clip(dh, -.1, .1)
        self.C += np.clip(dC, -.1, .1)
        # self.h += np.clip(dh, -10, 10)
        # self.C += np.clip(dC, -10, 10)

        # print "C", self.C
        # print "h", self.h
        # self.msg_motors.data.append(m[0])
        # self.msg_motors.data.append(m[1])
        # self.msg_motors.data = self.y[:,0].tolist()
        # print("sending msg", msg)
        # self.pub_motors.publish(self.msg_motors)
        # time.sleep(0.1)
        # if self.cnt > 20:
        #     rospy.signal_shutdown("stop")
        #     sys.exit(0)

    def local_hooks(self):
        pass
        
    def prepare_inputs(self):
        return self.robot.prepare_inputs()
    
    def prepare_output(self):
        return self.robot.prepare_output(self.y[:,0])

    def run(self):
        """LPZRos run method overwriting smp_thread_ros"""
        print("starting")
        while self.isrunning:
            # print "smp_thread: running"
            # call any local computations
            self.local_hooks()

            # prepare input for local conditions
            inputs = self.prepare_inputs()

            # execute network / controller
            self.brain(inputs)
            
            # local: adjust generic network output to local conditions
            self.prepare_output()

            # post hooks
            self.local_post_hooks()
            # write to memory
            # self.memory_pushback()
            
            # publish all state data
            # self.pub_all()
            
            # count
            self.cnt_main += 1 # incr counter
    
            # print "%s.run isrunning %d" % (self.__class__.__name__, self.isrunning) 
            
            # check if finished
            if self.cnt_main == 100000: # self.cfg.len_episode:
                # self.savelogs()
                self.isrunning = False
                # generates problem with batch mode
                rospy.signal_shutdown("ending")
                print("ending")
            
            self.rate.sleep()

# def get_config_sphero():
            
if __name__ == "__main__":
    import signal
    parser = argparse.ArgumentParser(description="lpzrobots ROS controller: test homeostatic/kinetic learning")
    parser.add_argument("-m",  "--mode",      type=str, help="select mode [hs] from " + str(LPZRos.modes), default = "hs")
    parser.add_argument("-lt", "--loop_time", type=float, help="loop time [1./20]", default=1./20)
    parser.add_argument("-r",  "--robot",     type=str,   help="Which robot [lpz], {lpz, sphero, puppy, roll-your-own}")
    args = parser.parse_args()

    # sanity check
    if not args.mode in LPZRos.modes:
        print "invalid mode string, use one of " + str(LPZRos.modes)
        sys.exit(0)
        
    lpzros = LPZRos(args.mode, loop_time = args.loop_time)

    def handler(signum, frame):
        print 'Signal handler called with signal', signum
        lpzros.isrunning = False
        sys.exit(0)
        # raise IOError("Couldn't open device!")

    # install interrupt handler
    signal.signal(signal.SIGINT, handler)

    
    lpzros.start()
    # prevent main from exiting
    while True and lpzros.isrunning:
        time.sleep(1)
        
    # rospy.spin()
    # while not rospy.shutdown():
