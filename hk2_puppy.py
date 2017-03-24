#!/usr/bin/env python
"""homekinesis with python, compare playfulmachines
$ python hk.py -h"""

# FIXME: put the learner / control structure into class to easily load
#        der/martius or reservoir model

# control: velocity and angle
# control: raw motors
# convenience: smp_thread_ros
# convenience: ros based setters for parameters
# convenience: publish preprocessed quantities

import time, argparse, sys
import numpy as np
import scipy.sparse as spa
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32, Float32MultiArray, ColorRGBA
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion #, Point, Pose, TwistWithCovariance, Vector3
from tiny_msgs.msg import tinyIMU
import cPickle as pickle
import os
import warnings

#from puppy_msgs.msg import puppy_maintenance

from smp_thread import smp_thread_ros
# from reservoirs import Reservoir

################################################################################
# helper funcs
def dtanh(x):
    return 1 - np.tanh(x)**2

def idtanh(x):
    return 1./dtanh(x) # hm?

################################################################################
# main homeostasis, homeokinesis class based on smp_thread_ros
class LPZRos(smp_thread_ros):
    modes = {"hs": 0, "hk": 1, "eh_pi_d": 2}
    control_modes = {"velocity" : 0, "position" : 1}

    def __init__(self, args):
        pubs = {
            "/homeostasis_motor": [Float32MultiArray,],
            "/lpzros/x": [Float32MultiArray],
            "/lpzros/xsi": [Float32MultiArray,],
            }
        subs = {
            "/tinyImu": [tinyIMU, self.cb_imu],
            }

        # gather arguments and transfer modes to integer representation
        self.mode = LPZRos.modes[args.mode]
        self.control_mode = LPZRos.control_modes[args.control_mode]
        self.numtimesteps = args.numtimesteps
        self.loop_time = args.looptime
        self.automaticMode = args.automaticMode
        self.verbose = args.verbose

        smp_thread_ros.__init__(self, loop_time = self.loop_time, pubs = pubs, subs = subs)

        # initialize counters
        self.cnt = 0
        self.cb_imu_cnt = 0


        ############################################################
        # model + meta params
        # self.numsen_raw = 11 # 5 # 2
        self.numsen = 6 # 5 # 2
        self.nummot = 4 # introduce 2 ghost variables
        self.imu_lin_acc_gain = 0 # 1e-3
        self.imu_gyrosco_gain = 1/5000.
        self.imu_orienta_gain = 0 # 1e-1
        self.linear_gain      = 0.0 # 1e-1
        self.output_gain = 32000 # 5000 # velocitycontrol

        self.msg_inputs     = Float32MultiArray()

        self.msg_xsi     = Float32MultiArray()
        self.msg_xsi.data = [0] * self.numsen

        self.msg_motors     = Float32MultiArray()
        self.msg_motors.data = [0] * self.nummot

        #self.msg_sensor_exp = Float64MultiArray()

        # sphero lag is 4 timesteps
        self.lag = 2 # 2 tested on puppy not so much influence
        # buffer size accomodates causal minimum 1 + lag time steps
        self.bufsize = 1 + self.lag # 2
        self.creativity = args.creativity
        # self.epsA = 0.1
        # self.epsA = 0.02
        self.epsA = args.epsA
        # self.epsC = 0.001
        #self.epsC = 0.01
        #self.epsC = 0.9
        self.epsC = args.epsC

        # sampling scale for automaticMode
        self.automaticModeScaleEpsA = 0.5
        self.automaticModeScaleEpsC = 0.5

        if self.automaticMode:
            self.epsA = np.random.exponential(scale=self.automaticModeScaleEpsA, size = 1)[0]
            self.epsC = np.random.exponential(scale=self.automaticModeScaleEpsC, size = 1)[0]

        print "EpsA:\t%f\nEpsC:\t%f\nCreativity:\t%f\nEpisodelength:\t%d\nLag:\t%d" % (self.epsA, self.epsC,self.creativity, self.numtimesteps,self.lag)

        ############################################################
        # forward model
        self.A  = np.random.uniform(-1e-1, 1e-1, (self.numsen, self.nummot))
        self.b = np.zeros((self.numsen,1))

        print "initial A"
        print self.A

        # controller
        self.C  = np.random.uniform(-1e-1, 1e-1, (self.nummot, self.numsen))

        print "initial C"
        print self.C

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

        self.xsiAvg = 0
        self.xsiAvgSmooth = 0.01

        self.imu  = tinyIMU()
        self.imu_vec  = np.zeros((3 + 3, 1))
        self.imu_smooth = 0.5 # coef

        self.motorSmooth = 0.5

        warnings.filterwarnings("error")
        np.seterr(all='print')
        self.exceptionCounter = 0
        self.maxExceptionCounter = 10


        # If something is changed in the variableDict style or content, change dataversion!
        # Version History:
        # 1: Initial Commit
        # 2: xsi was not saved correctly before.
        # 3: lag and automaticMode + scales included
        # 4: exceptionCounter added
        # 5: exceptionCounter working :) forgot to increase versionnumber and wasn't working good
        # 6: added file id (when there are multiple results with the same eps and c) and filename

        self.variableDict = {
            "dataversion": 5,
            "timesteps" : self.numtimesteps,
            "looptime" : self.loop_time,
            "startTime" : time.time(),
            "endTime" : 0,
            "numsen": self.numsen,
            "nummot": self.nummot,
            "epsA": self.epsA,
            "epsC": self.epsC,
            "creativity": self.creativity,
            "imuSmooth": self.imu_smooth,
            "motorSmooth": self.motorSmooth,
            "lag": self.lag,
            "automaticMode": self.automaticMode,
            "automaticModeScaleEpsA": self.automaticModeScaleEpsA,
            "automaticModeScaleEpsC": self.automaticModeScaleEpsC,
            "exceptionCounter": 0,
            "id": 0,
            "filename": 0,
            "C": np.zeros((self.numtimesteps,) + self.C.shape),
            "A": np.zeros((self.numtimesteps,) + self.A.shape),
            "h": np.zeros((self.numtimesteps,) + self.h.shape),
            "b": np.zeros((self.numtimesteps,) + self.b.shape),
            "x": np.zeros((self.numtimesteps,) + self.x.shape),
            "y": np.zeros((self.numtimesteps,) + self.y.shape),
            "xsi": np.zeros((self.numtimesteps,) + self.xsi.shape)
        }



        # expansion
        # self.exp_size = self.numsen
        # self.exp_hidden_size = 100
        # self.res = Reservoir(N = self.exp_hidden_size, p = 0.1, g = 1.5, tau = 0.1, input_num = self.numsen_raw, input_scale = 5.0)
        # self.res_wo_expand     = np.random.randint(0, self.exp_hidden_size, self.exp_size)
        # self.res_wo_expand_amp = np.random.uniform(0, 1, (self.exp_size, 1)) * 0.8
        # self.res_wi_expand_amp = np.random.uniform(0, 1, (self.exp_size, self.numsen_raw)) * 1.0

    # def expansion_random_system(self, x, dim_target = 1):
    #     # dim_source = x.shape[0]
    #     # print "x", x.shape
    #     self.res.execute(x)
    #     # print "self.res.r", self.res.r.shape
    #     a = self.res.r[self.res_wo_expand]
    #     # print "a.shape", a.shape
    #     b = a * self.res_wo_expand_amp
    #     # print "b.shape", b.shape
    #     c = b + np.dot(self.res_wi_expand_amp, x)
    #     return c


    def cb_imu(self, msg):
        """ROS IMU callback: use odometry and incoming imu data to trigger
        sensorimotor loop execution"""

        self.imu = msg
        imu_vec_acc = np.array((self.imu.accel.x, self.imu.accel.y, self.imu.accel.z))
        imu_vec_gyr = np.array((self.imu.gyro.x, self.imu.gyro.y, self.imu.gyro.z))

        imu_vec_ = np.hstack((imu_vec_acc, imu_vec_gyr)).reshape(self.imu_vec.shape)
        #print "imu_vec_", imu_vec_
        #self.imu_smooth = (self.y[5,0] + 1) / 2
        self.imu_smooth = 0.5
        self.imu_vec = self.imu_vec * self.imu_smooth + (1 - self.imu_smooth) * imu_vec_

        self.cb_imu_cnt += 1

    def algorithm_learning_step(self, msg):
        try:
            """lpz sensors callback: receive sensor values, sos algorithm attached"""
            # FIXME: fix the timing
            now = 0
            # self.msg_motors.data = []
            self.x = np.roll(self.x, 1, axis=1) # push back past
            self.y = np.roll(self.y, 1, axis=1) # push back past
            # update with new sensor data
            self.x[:,now] = np.array(msg).reshape((self.numsen, ))
            self.msg_inputs.data = self.x[:,now].flatten().tolist()
            self.pub["_lpzros_x"].publish(self.msg_inputs)

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
            # print "m1.shape", m1.shape, m1
            t1 = self.g(m1 + self.h).reshape((self.nummot,))
            self.y[:,now] = t1
            # print "t1", t1
            self.cnt += 1
            if self.cnt <= 2: return

            # print "x", self.x
            # print "y", self.y

            # local variables
            x = np.atleast_2d(self.x[:,self.lag]).T
            # print(x.flatten())
            # this is wrong
            # y = np.atleast_2d(self.y[:,self.lag]).T
            # this is better
            y = np.atleast_2d(self.y[:,self.lag]).T
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
            xsi = x_fut - (np.dot(self.A, y) + self.b)

            # for the purpose of pickling it later
            self.xsi = xsi

            self.xsiAvg = np.sum(np.abs(xsi)) * self.xsiAvgSmooth + (1 - self.xsiAvgSmooth) * self.xsiAvg

            self.msg_xsi.data = xsi.flatten().tolist()
            self.msg_xsi.data.append(self.xsiAvg)
            self.pub["_lpzros_xsi"].publish(self.msg_xsi)

            if(self.verbose): print("Xsi Average %f" % self.xsiAvg)

            # forward model learning
            self.A += self.epsA * np.dot(xsi, y.T) + (self.A * -0.003) * 0.1
            # self.A += self.epsA * np.dot(xsi, np.atleast_2d(self.y[:,0])) + (self.A * -0.003) * 0.1
            self.b += self.epsA * xsi              + (self.b * -0.001) * 0.1

            # print "A", self.cnt, self.A
            # print "b", self.b

            if self.mode == 1: # TLE / homekinesis
                eta = np.dot(np.linalg.pinv(self.A), xsi)
                zeta = np.clip(eta * g_prime_inv, -1., 1.)
                if self.verbose: print "eta", self.cnt, eta
                if self.verbose: print "zeta", self.cnt, zeta
                # print "C C^T", np.dot(self.C, self.C.T)
                # mue = np.dot(np.linalg.pinv(np.dot(self.C, self.C.T)), zeta)
                lambda_ = np.eye(self.nummot) * np.random.uniform(-0.01, 0.01, self.nummot)
                mue = np.dot(np.linalg.pinv(np.dot(self.C, self.C.T) + lambda_), zeta)
                v = np.clip(np.dot(self.C.T, mue), -1., 1.)
                self.v_avg += (v - self.v_avg) * 0.1
                if self.verbose: print "v", self.cnt, v
                if self.verbose: print "v_avg", self.cnt, self.v_avg
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


            elif self.mode == 0: # homestastic learning
                eta = np.dot(self.A.T, xsi)
                if self.verbose: print "eta", self.cnt, eta.shape, eta
                dC = np.dot(eta * g_prime, x.T) * self.epsC
                dh = eta * g_prime * self.epsC
                # print dC, dh
                # self.C +=

            # FIXME: ???
            self.h += np.clip(dh, -.1, .1)
            self.C += np.clip(dC, -.1, .1)

            if self.verbose: print "C:\n" + str(self.C)
            if self.verbose: print "A:\n" + str(self.A)

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
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly,
            self.exceptionCounter += 1
            print "Error number %d" %( self.exceptionCounter)

            # if there are too many exceptions, end experiment
            if(self.exceptionCounter > self.maxExceptionCounter):
                print "Experiment forced to quit"
                self.cnt_main = self.numtimesteps - 1


    def prepare_inputs(self):
        #imu_smooth = {1, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05}

        inputs = (
            self.imu_vec[0] * self.imu_gyrosco_gain,
            self.imu_vec[1] * self.imu_gyrosco_gain,
            self.imu_vec[2] * self.imu_gyrosco_gain,
            self.imu_vec[3] * self.imu_gyrosco_gain,
            self.imu_vec[4] * self.imu_gyrosco_gain,
            self.imu_vec[5] * self.imu_gyrosco_gain,
            #self.msg_motors.data[0] / self.output_gain,
            #self.msg_motors.data[1] / self.output_gain,
            #self.msg_motors.data[2] / self.output_gain,
            #self.msg_motors.data[3] / self.output_gain,
            #self.y[0,0],
            #self.y[1,0],
            #self.y[2,0],
            #self.y[3,0],
        )

        # check input dimensionality
        if(len(inputs) != self.numsen):
            raise Exception("numsen doesn't match up with the real input data dimensionality numsen: " + str(self.numsen) + ", len: " + str(len(inputs)))

        if self.verbose: print "Inputs: ", inputs
        return inputs

    def prepare_and_send_output(self):

        # velocity control
        if self.control_mode == 0:
            motor_old = np.array(self.msg_motors.data)
            self.msg_motors.data = np.clip((motor_old + self.y[:,0]) * self.output_gain, -32000, 32000).tolist()

        # position control
        elif self.control_mode == 1:
            motor_old = np.array(self.msg_motors.data)

            #self.motorAlpha = (self.y[4,0] + 1)/ 2. * 0.7 + 0.1 # between 0.1 and 0.8
            self.msg_motors.data = self.motorSmooth * self.y[:,0] * self.output_gain + (1 - self.motorSmooth) * motor_old
        else:
            raise Exception("Unknown control mode " + str(self.control_mode))

        if self.verbose: print "y = %s , motors = %s" % (self.y[:,0], self.msg_motors)

        self.pub["_homeostasis_motor"].publish(self.msg_motors)

    def run(self):
        """LPZRos run method overwriting smp_thread_ros"""
        print("starting")
        while self.isrunning:
            # print "smp_thread: running"

            # prepare input for local conditions
            inputs = self.prepare_inputs()

            # execute network / controller
            self.algorithm_learning_step(inputs)

            # adjust generic network output to local conditions and send
            self.prepare_and_send_output()

            # store all variables for logging
            self.storeAllVariables()

            # count
            self.cnt_main += 1 # incr counter

            # check if finished
            if self.cnt_main == self.numtimesteps: # self.cfg.len_episode:
                # self.savelogs()
                self.isrunning = False

                # write end time
                self.variableDict["endTime"] = time.time()
                self.variableDict["exceptionCounter"] = self.exceptionCounter

                # save dict to file
                self.pickleDumpVariables()

                # generates problem with batch mode
                rospy.signal_shutdown("ending")
                print("ending")

            self.rate.sleep()

    def storeAllVariables(self):
        self.storeVariable("C", self.C)
        self.storeVariable("A", self.A)
        self.storeVariable("h", self.h)
        self.storeVariable("b", self.b)
        self.storeVariable("x", self.x)
        self.storeVariable("y", self.y)
        self.storeVariable("xsi", self.xsi)

    def storeVariable(self, name, value):
        self.variableDict[name][self.cnt_main,:,:] = value

    def pickleDumpVariables(self):
        id = 0
        while True:
            filename = self.combineName("pickles/", self.epsC, self.epsA, self.creativity, self.numtimesteps, id, ".pickle")

            if not os.path.exists(filename):
                self.variableDict["id"] = id
                self.variableDict["filename"] = filename
                pickle.dump(self.variableDict, open(filename, "wb"))
                pickle.dump(self.variableDict, open("pickles/newest.pickle", "wb"))
                print "pickle saved: %s and pickles/newest.pickle" % (filename)
                return

            # increment id and try again
            id += 1
            if id == 10000:
                raise Exception("While searching for an id to save the pickle 10000 was reached. Did you really do so many?")


    def combineName(self, prefix, epsC, epsA, creativity, timesteps, id, postfix):
        #return prefix + "recording_eC" + str(epsC) + "_eA" + str(epsA) + "_c" + str(creativity) + "_n" + str(timesteps) + "_id" + str(id) + postfix
        filename= "%srecording_eC%.2f_eA%.2f_c%.2f_n%d_id%d%s" % (prefix, epsC, epsA, creativity, timesteps, id, postfix)
        if self.verbose: print filename
        return filename

if __name__ == "__main__":
    import signal
    parser = argparse.ArgumentParser(description="lpzrobots ROS controller: test homeostatic/kinetic learning")
    parser.add_argument("-m", "--mode", type=str, help="select mode [hs] from " + str(LPZRos.modes), default = "hk")
    parser.add_argument("-cm", "--control_mode", type=str, help="select control mode from " + str(LPZRos.control_modes), default="position")
    parser.add_argument("-n", "--numtimesteps", type=int, help="Episode length in timesteps, standard 1000", default= 1000)
    parser.add_argument("-lt", "--looptime", type=float, help="delay between timesteps in the loop", default = 0.05)
    parser.add_argument("-eC", "--epsC", type=float, help="learning rate for controller", default = 0.2)
    parser.add_argument("-eA", "--epsA", type=float, help="learning rate for model", default = 0.01)
    parser.add_argument("-c", "--creativity", type=float, help="creativity", default = 0.5)
    parser.add_argument("-auto", "--automaticMode", type=bool, help="draw parameters from random sample", default=False)
    parser.add_argument("-v", "--verbose", type=bool, help="print many motor and sensor commands", default=False)

    args = parser.parse_args()

    # sanity check
    if not args.mode in LPZRos.modes:
        print "invalid mode string, use one of " + str(LPZRos.modes)
        sys.exit(0)

    if not args.control_mode in LPZRos.control_modes:
        print "invalid control_mode string, use one of " + str(LPZRos.control_modes)
        sys.exit(0)

    lpzros = LPZRos(args)

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
