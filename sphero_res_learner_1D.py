#!/usr/bin/env python
"""self-organizing behaviour: smp / inverse model learning for sphero, 1-dimensional"""

import signal, time, sys, argparse, os

import numpy as np
import numpy.linalg as LA

# # additional paths
# localpaths = ["/path/to/smp/smp"]
# if localpaths[0] not in sys.path:
#     sys.path.insert(0, localpaths[0])
    
from smp_base.models_reservoirs import Reservoir
from smp_base.models_learners import learnerEH
from smp_base.smp_thread import smp_thread_ros

import rospy
from std_msgs.msg import Float32, Float32MultiArray, ColorRGBA
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion #, Point, Pose, TwistWithCovariance, Vector3
from smp_msgs.msg import reservoir
import tf

class SpheroResLearner1D(learnerEH, smp_thread_ros):
    """Sphero experiments"""
    modes = {"vel": 0, "res_gen": 1}
    def __init__(self, args):
        smp_thread_ros.__init__(self)
        learnerEH.__init__(self, args)
        self.name = "SpheroResLearner1D"
        # print mode
        self.mode = SpheroResLearner1D.modes[args.mode]
        # self.loop_time = 1/100.
        # self.loop_time = 1/50.
        # self.loop_time = 1/40.
        self.loop_time = 1/20.
        # self.loop_time = 1/10.
        # self.loop_time = 1.
        self.isrunning = True
        # rospy.init_node(self.name)
        self.sub["imu"] = rospy.Subscriber("/imu",
                                        Imu, self.cb_imu)
        self.sub["odom"] = rospy.Subscriber("/odom",
                                        Odometry, self.cb_odom)
        if os.environ["ROS_DISTRO"] == "hydro":
            self.pub["twist"] = rospy.Publisher("/cmd_vel", Twist)
            self.pub["color"] = rospy.Publisher("/set_color", ColorRGBA)
            self.pub["target"] = rospy.Publisher("/learner/target", reservoir)
            self.pub["learn_zn"] = rospy.Publisher("/learner/zn", reservoir)
        else:
            self.pub["twist"] = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
            self.pub["color"] = rospy.Publisher("/set_color", ColorRGBA, queue_size=1)
            self.pub["target"] = rospy.Publisher("/learner/target", reservoir, queue_size=1)
            self.pub["learn_zn"] = rospy.Publisher("/learner/zn", reservoir, queue_size=1)
        # counting
        self.cb_odom_cnt = 0
        self.cb_imu_cnt = 0
        self.cnt = 0
        self.cnt_inc = 1
        
        # save odometry state
        self.odom = Odometry()
        # motor output struct
        self.T = Twist()
        print("Twist", self.T)
        
        self.err = np.zeros((self.cfg.len_episode, self.cfg.odim))
        # self.cfg.target = np.zeros_like(self.iosm.x)
        self.cfg.target = np.ones((self.cfg.odim, 1)) * -0.6
        # self.target_change_period = np.random.uniform(20, 100) * 10
        # self.target_change_period = np.random.uniform(20, 100) * 1
        self.target_change_period = 100
        
        self.noise = 0.

        # sphero color
        self.color = ColorRGBA()

        # learning
        self.len_learning = self.cfg.len_episode * self.cfg.ratio_testing
        
    def cb_odom(self, msg):
        """ROS odometry callback, copy incoming data into local memory"""
        # print type(msg)
        self.odom = msg
        # self.iosm.x[0] = self.odom.pose.pose.position.x
        # self.iosm.x[1] = self.odom.pose.pose.position.y
        # self.iosm.x_raw[0] = self.odom.pose.pose.position.x
        # self.iosm.x_raw[1] = self.odom.pose.pose.position.y
        self.iosm.x_raw[0] = self.odom.twist.twist.linear.x
        # self.iosm.x_raw[1] = self.odom.twist.twist.linear.x - self.cfg.target
        
        # self.iosm.x_raw[3] = self.odom.twist.twist.linear.y
        self.cb_odom_cnt += 1
        return

    def cb_imu(self, msg):
        """ROS IMU callback: use odometry and incoming imu data to trigger
        sensorimotor loop execution"""
        # print "odom", self.odom
        # print "imu", msg
        # return

        # (r, p, y) = tf.transformations.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        # # print r, p, y
        # self.iosm.x_raw[4] = r
        # self.iosm.x_raw[5] = p
        # self.iosm.x_raw[6] = y
        # self.iosm.x_raw[7] = msg.angular_velocity.x
        # self.iosm.x_raw[8] = msg.angular_velocity.y
        # self.iosm.x_raw[9] = msg.angular_velocity.z
        # self.iosm.x_raw[10] = msg.linear_acceleration.x
        # self.iosm.x_raw[11] = msg.linear_acceleration.y
        # self.iosm.x_raw[12] = msg.linear_acceleration.z

        self.cb_imu_cnt += 1
        # time.sleep(0.1)
        # if self.cnt > 20:
        #     rospy.signal_shutdown("stop")
        #     sys.exit(0)
        
    def local_hooks(self):
        pass
    
    def prepare_inputs(self):
        if False and self.use_icm: # always true as of learnerEH 20150304
            # print (self.iosm.x_raw)
            self.iosm.x = self.input_coupling_mtx * self.iosm.x_raw # component-wise
        else:
            self.iosm.x[1,0] = self.iosm.x_raw[0,0] - self.cfg.target[0,0]
            # self.iosm.x[1,0] = self.cfg.target[0,0]
            self.iosm.x[0,0] = self.iosm.x_raw[0,0]
            
        # self.iosm.x = self.iosm.x_raw
        # self.iosm.x[0:2,0] = 0.
        # self.iosm.x[4:,0] = 0.
        print("self.iosm.x", self.iosm.x)
        # pass
    
    def prepare_output(self, z, zn):
        print("z, zn", z, zn)
        if self.cfg.mode == "vel":
            if self.cnt_main < self.len_learning:
                # reassign
                z_ = zn[0]
            else:
                # reassign
                z_ = z[0]
            self.T.linear.x = (0. + z_) * self.cfg.res_output_scaling
            
        else:
            self.T.linear.x = (0.0 + zn[0]) * self.cfg.res_output_scaling
            
        # self.T.linear.y = zn[1] * self.cfg.res_output_scaling

        # set sphero color
        # if z >= 0.:
        #     self.color.r = z
        # else:
        #     self.color.g = np.abs(z)
        # print "color: error", self.iosm.e
        color = np.clip(np.abs(self.iosm.e), 0., 1.)
        self.color.r = color
        self.pub["color"].publish(self.color)
        self.pub["twist"].publish(self.T)

    def controller(self):
        # self.T.angular.x = np.random.uniform(0.0, 0.1)
        now = self.cnt_main
        print("iosm.x_raw", now, self.iosm.x_raw)
        # print "now", now
        if self.cfg.mode == "vel":
            if True: # self.cnt_main % 1 == 0:
                # target_change_period = 300
                # target_change_period = 100.
                # target_change_period = 50.
                # target_change_period = 20.
                target_change_period = self.target_change_period
                print("xxxxx", self.cfg.tp_target_spec)
                # constant target
                if self.cfg.tp_target_spec["constant"]:
                    if self.cnt_main == 10:
                        # self.target[0,0] = self.cfg.target
                        if np.random.uniform(0, 1.) > 0.5:
                            target_sign = 1
                        else:
                            target_sign = -1
                        self.cfg.target[0,0] = target_sign * np.random.uniform(0.5, 1.2)
                    
                elif self.cfg.tp_target_spec["jumping"]:
                    # jumping target
                    print(list(range(self.cfg.len_washout+1, self.cfg.len_episode, int(target_change_period))))
                    if self.cnt_main in range(self.cfg.len_washout+1, self.cfg.len_episode, int(target_change_period)):
                        self.cfg.target[0,0] = 1. + np.random.uniform(-0.5, 0.5)
                        # self.cfg.target[0,0] = np.random.uniform(-1, 1)
                    
                elif self.cfg.tp_target_spec["jumping_sign"]:
                    # jumping target inverting sign
                    if self.cnt_main in range(self.cfg.len_washout+1, self.cfg.len_episode, int(target_change_period+1)):
                        if self.cnt_main % 2 == 0:
                            print("even")
                            self.cfg.target[0,0] = 1. + np.random.uniform(-0.3, 0.3)
                        else:
                            print("odd")
                            self.cfg.target[0,0] = -1. + np.random.uniform(-0.3, 0.3)

                elif self.cfg.tp_target_spec["sine"]:
                    # sinewave target fixed amp
                    print("sinewave target")
                    # self.cfg.target[0,0] = -1.25 + 0.5 * np.cos(self.cnt_main/15.)
                    # self.cfg.target[0,0] = 1 + 0.3 * np.sin(self.cnt_main/target_change_period)
                    self.cfg.target[0,0] = 1. + 0.3 * np.sin(self.cnt_main/float(target_change_period))
                    print("self.cfg.target", self.cfg.target)
                elif self.cfg.tp_target_spec["sine_sign"]:
                    # sinewave target fixed amp
                    # self.cfg.target[0,0] = -1.25 + 0.5 * np.cos(self.cnt_main/15.)
                    # self.cfg.target[0,0] = 1 + 0.3 * np.sin(self.cnt_main/target_change_period)
                    self.cfg.target[0,0] = 0. + 1. * np.sin(self.cnt_main/target_change_period)
    
                # # sinewave with increasing amplitude
                # self.cfg.target[0,0] = 1.
                # sinamp = np.clip(self.cnt_main/4000., -0.2, 0.2)
                # self.cfg.target[0,0] += sinamp * np.sin(self.cnt_main/50.)

                print("target", self.cfg.target)
                self.pub["target"].publish(self.cfg.target)
                err = self.iosm.x_raw[0] - self.cfg.target #
                # print "err", err
                err1 = -np.sum(np.square(err))
                self.err[self.cnt_main,0] = err1
                if self.cnt_main > 200:
                    # print "mse window", self.err[(self.cnt_main-200):self.cnt_main,0]
                    self.iosm.mse = np.mean(np.sqrt(-self.err[(self.cnt_main-200):self.cnt_main,0])) * np.ones_like(self.cfg.target)
                    print("mse", self.iosm.mse)
                else:
                    self.iosm.mse = np.mean(np.sqrt(-self.err[0:self.cnt_main,0])) * np.ones_like(self.cfg.target)                    
                # self.err[self.cnt_main,1] = err1
                # print "self.rew.perf.shape", self.rew.perf.shape
                # print self.err[self.cnt_main,0]
                # FIXED: acceleration based reward
                # FIXME: sign is inverted when running on different machines?
                # FIXME: this performance measure asks for infinitely large accelerations
                # self.rew.perf = -np.sign(err) * np.array([self.iosm.x_raw[2,0]]).reshape((1,1)) # eta
                # self.rew.perf[0,0] = err1 # np.sign(err) * np.array([self.iosm.x_raw[2,0]]).reshape((1,1)) # rudi
                # FIXME: take derivative of error
                self.rew.perf[0,0] = err1 + self.iosm.e[0,0]
                # self.rew.perf[1,0] = err1

                self.iosm.e[0,0] = err1
                self.iosm.t[0,0] = self.cfg.target[0,0]
                
                # if np.abs(self.rew.perf) > 1:
                #     self.rew.perf = np.array((-1.)).reshape((1,1))
                
                # lp history sould be decaying actually, so it's easier to be better
                self.rew.perf_lp = ((1 - self.rew.coeff_a) * self.rew.perf_lp) + (self.rew.coeff_a * self.rew.perf)
                if self.cfg.use_et:
                    self.learnEHE()
                    # pass
                else:
                    self.learnEH()
                    # pass
                # self.learnEHpub()
                # energy based reward
                # learn stuff
                # self.iosm.x[1] = err
                # self.iosm.x[1] = self.cfg.target[0,0] - 1.
                # self.iosm.x[1] = 0.
                self.iosm.z = self.res.execute(self.iosm.x)
                self.iosm.zn = self.res.zn
                # self.iosm.z = self.res.execute(err)
                # self.iosm.z[1,0] = 0.
                # self.res.z[1,0] = 0.
                # self.res.zn[1,0] = 0.
                # reset input
                # self.iosm.x_raw[0,0] = 0.
            else:
                self.iosm.z = self.iosm.z
            # return (self.res.z, self.res.zn)
            # return (self.iosm.z, self.iosm.zn)
        elif self.cfg.mode == "PIDvel":
            self.err = self.iosm.x_raw[2:4] - np.ones((self.cfg.odim, 1)) * 0.2
            print("self.err", self.err)
            # self.iosm.z = 10.0 * self.err
            self.iosm.z[0,0] = 0.5 * np.cos(self.cnt_main/20.)
            # self.iosm.z[1,0] = 0.5 * np.sin(self.cnt_main/10.)
        elif self.cfg.mode == "noise":
            noise = np.random.normal(0.5, 1.)
            self.noise = 0.9 * self.noise + 0.1 * noise
            # self.noise = noise
            # self.noise = 1. + 0.5 * np.sin(self.cnt_main/10.)
            # self.noise = 0.7
            self.iosm.z[0,0] = self.noise
        elif self.cfg.mode == "bump":
            print("bump")
            # z = np.zeros((self.cfg.odim, 1))
            if self.cnt_main in range(100, 200):
                self.iosm.z[0,0] = 1.
            elif self.cnt_main in range(300, 400):
                self.iosm.z[0,0] = -1.
            # elif self.cnt_main in range(260, 280):
            #     self.iosm.z[1,0] = 1.
            # elif self.cnt_main in range(320, 340):
            #     self.iosm.z[1,0] = -1.
            else:
                self.iosm.z[0,0] = 0.
                # self.iosm.z[1,0] = 0.
        elif self.cfg.mode == "ramp":
            if self.cnt >= 255:
                self.cnt_inc = -1
            elif self.cnt <= -255:
                self.cnt_inc = 1
            self.cnt += self.cnt_inc
            self.iosm.zn[0,0] = self.cnt / 100.
            print("ramp", self.iosm.zn)
            # self.iosm.z[0,0] = (255 + np.abs((self.cnt_main % 510) - 255)) * 0.005
            # if np.random.uniform(0, 1.) > 0.9:
            #     print "hickup"
            #     time.sleep(0.5)

        # store states for pushback: see above

        print("publishing zn")
        self.pub["learn_zn"].publish(self.iosm.zn)
        return(self.iosm.z, self.iosm.zn)


def main(args):
    # fix RNG seed
    np.random.seed(args.seed)

    # check datadir exists
    if not os.path.exists(args.datadir):
        print("Datadir doesn't exist, try mkdir '%s'" % (args.datadir))
    
    if args.mode == "res_gen":
        # pre-generate N reservoirs for use in experiments
        for i in range(10):
            s = SpheroResLearner1D(args)
            timestamp = time.strftime("%Y-%m-%d-%H%M%S")
            filename = "%s/reservoir-%d-%s.bin" % (args.datadir, i, timestamp)
            s.res.save(filename=filename)
        s.isrunning = False
        sys.exit(0)
    else:
        s = SpheroResLearner1D(args)
        # do we get passed a trained network for testing?
        if args.resfile != "":
            # then load it
            print("loading reservoir from file %s" % (args.resfile))
            s.res.load(filename="%s/%s" % (args.datadir, args.resfile))
            # bend some parameters, e.g. eta and theta
            print("overriding config for eta, theta")
            s.cfg.eta_EH = 0.
            s.cfg.res_theta = 1e-3
            s.res.set_theta(s.cfg.res_theta)
    
    def handler(signum, frame):
        print('Signal handler called with signal', signum)
        s.isrunning = False
        sys.exit(0)
        # raise IOError("Couldn't open device!")

    # install interrupt handler
    signal.signal(signal.SIGINT, handler)

    # run
    s.start()
    
    while not rospy.is_shutdown():
        # print("main loop")
        time.sleep(1)

    # timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    target_str = list(s.cfg.tp_target_spec.keys())[list(s.cfg.tp_target_spec.values()).index(1)]
    print("target_str", target_str)
    filename = "%s/log-learner-%s-N-%d-eta-%f-theta-%f-g-%f-target-%s" % (args.datadir,
                                                                                   timestamp,
                                           s.cfg.N, s.cfg.eta_EH, s.cfg.res_theta, s.cfg.g,
                                           target_str)
    # save logs
    s.savelogs(ts=timestamp, filename=filename)
    # save network
    if args.resfile == "":
        target_str = list(s.cfg.tp_target_spec.keys())[list(s.cfg.tp_target_spec.values()).index(1)]
        print("target_str", target_str)
        filename = "%s/reservoir-%d-%s-N-%d-eta-%f-theta-%f-g-%f-target-%s.bin" % (args.datadir,
                                                                                   1, timestamp,
                                           s.cfg.N, s.cfg.eta_EH, s.cfg.res_theta, s.cfg.g,
                                           target_str)
        s.res.save(filename=filename)
    else:
        print("not saving network, already loaded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reservoir smp learner for sphero")
    parser.add_argument("-c", "--config", dest="cfgfilename", help="config file to use",
                        default="default.cfg")
    parser.add_argument("-d", "--datadir", help="Directory root for saved configurations",
                        default=parser.prog[0:-3])
    parser.add_argument("-m", "--mode", type=str, help="select mode: " + str(SpheroResLearner1D.modes),
                        default="vel")
    parser.add_argument("-rf", "--resfile", dest="resfile", help="pickled reservoir to load",
                        default="")
    parser.add_argument("-s", "--seed", dest="seed", help="seed for rng", default=123, type=int)
    args = parser.parse_args()

    # good seeds: 4, 
    
    main(args)
