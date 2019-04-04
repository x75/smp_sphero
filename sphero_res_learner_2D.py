#!/usr/bin/env python
"""self-organizing behaviour: smp / inverse model learning for sphero, 2-dimensional"""

import rospy
import signal
import time, sys, argparse, os

from std_msgs.msg import Float32, Float32MultiArray, ColorRGBA
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion #, Point, Pose, TwistWithCovariance, Vector3
from smp_msgs.msg import reservoir
import tf

# # additional paths
# localpaths = ["/path/to/smp/smp"]
# if localpaths[0] not in sys.path:
#     sys.path.insert(0, localpaths[0])

from smp_base.models_reservoirs import Reservoir
from smp_base.models_learners import learnerEH
from smp_base.smp_thread import smp_thread_ros

import numpy as np
import numpy.linalg as LA

class SpheroResLearner2D(learnerEH, smp_thread_ros):
    """Sphero experiments"""
    modes = {"vel": 0, "res_gen": 1}
    def __init__(self, args):
        smp_thread_ros.__init__(self)
        learnerEH.__init__(self, args)
        self.name = "SpheroResLearner2D"
        # print mode
        self.mode = SpheroResLearner2D.modes[args.mode]
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
        # self.pub["twist"] = rospy.Publisher("/cmd_vel", Twist)
        # self.pub["color"] = rospy.Publisher("/set_color", ColorRGBA)
        # self.pub["target"] = rospy.Publisher("/learner/target", reservoir)
        # self.pub["learn_zn"] = rospy.Publisher("/learner/zn", reservoir)
        if os.environ["ROS_DISTRO"] == "hydro":
            self.pub["twist"] = rospy.Publisher("/cmd_vel", Twist)
            self.pub["color"] = rospy.Publisher("/set_color", ColorRGBA)
            self.pub["target"] = rospy.Publisher("/learner/target", reservoir)
            self.pub["learn_zn"] = rospy.Publisher("/learner/zn", reservoir)
        else:
            self.pub["twist"] = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
            self.pub["twist_raw"] = rospy.Publisher("/cmd_vel_raw", Twist, queue_size=1)
            self.pub["color"] = rospy.Publisher("/set_color", ColorRGBA, queue_size=1)
            self.pub["target"] = rospy.Publisher("/learner/target", reservoir, queue_size=1)
            self.pub["learn_zn"] = rospy.Publisher("/learner/zn", reservoir, queue_size=1)
        # counting
        self.cb_odom_cnt = 0
        self.cb_imu_cnt = 0
        self.cnt = 0

        # save odometry state
        self.odom = Odometry()
        # motor output struct
        self.T = Twist()
        print "Twist", self.T
        
        self.err = np.zeros((self.cfg.len_episode, self.cfg.odim))
        # self.cfg.target = np.zeros_like(self.iosm.x)
        self.cfg.target = np.ones((self.cfg.odim, 1)) * -0.6
        # this  is the same as self.cfg.target_interval
        # self.target_change_period = np.random.uniform(20, 60) # * 10
        self.target_change_period = 100.
        self.real_target = np.zeros_like(self.cfg.target)
        self.phi = 0.15
        self.phi_inc = 1/100.
        # self.phi_inc = 1/20.
        self.vel = 0.8
        self.target_jumps = (self.cfg.len_episode * (-np.logspace(1, 0, 10)+10)/9.).astype(int)

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
        self.iosm.x_raw[1] = self.odom.twist.twist.linear.y
        # self.iosm.x_raw[2] = self.odom.twist.twist.linear.y
        # self.iosm.x_raw[3] = self.odom.twist.twist.linear.y
        self.cb_odom_cnt += 1
        return

    def cb_imu(self, msg):
        """ROS IMU callback: use odometry and incoming imu data to trigger
        sensorimotor loop execution"""
        # print "odom", self.odom
        # print "imu", msg
        # return

        (r, p, y) = tf.transformations.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        # print "r, p, y", r, p, y
        print "r", r
        self.iosm.x_raw[2] = r
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
            # x_tmp = self.iosm.x_raw[0,0] * np.cos(self.iosm.x_raw[2,0])
            # y_tmp = self.iosm.x_raw[0,0] * np.sin(self.iosm.x_raw[2,0])
            x_tmp = self.iosm.x_raw[0,0]
            y_tmp = self.iosm.x_raw[1,0]
            self.iosm.x[0,0] = x_tmp - self.cfg.target[0,0]
            self.iosm.x[2,0] = y_tmp - self.cfg.target[1,0]
            # self.iosm.x[0,0] = self.cfg.target[0,0]
            # self.iosm.x[2,0] = self.cfg.target[1,0]
            # self.iosm.x[2,0] = y_tmp - self.cfg.target[1,0]
            self.iosm.x[1,0] = x_tmp
            self.iosm.x[3,0] = y_tmp
        # self.iosm.x = self.iosm.x_raw
        # self.iosm.x[0:2,0] = 0.
        # self.iosm.x[4:,0] = 0.
        print "self.iosm.x", self.iosm.x
        # pass
    
    def prepare_output(self, z, zn):
        print "z, zn", z, zn
        if self.cfg.mode == "vel":
            if self.cnt_main < self.len_learning:
                # z_ = zn[0]
                z_ = zn
            else:
                # z_ = z[0]
                z_ = z
            self.T.linear.x = (0. + z_[0]) * self.cfg.res_output_scaling
            # self.T.angular.z = # (0. + z_[0]) * self.cfg.res_output_scaling
            self.T.linear.y = (0. + z_[1]) * self.cfg.res_output_scaling
        else:
            self.T.linear.x = (0.0 + zn[0]) * self.cfg.res_output_scaling
            self.T.linear.y = (0.0 + zn[1]) * self.cfg.res_output_scaling
        # self.T.linear.y = zn[1] * self.cfg.res_output_scaling
        # print "Twist", z_, self.T

        # set sphero color
        # if z >= 0.:
        #     self.color.r = z
        # else:
        #     self.color.g = np.abs(z)
        # print "color: error", self.iosm.e
        color = np.clip(np.sum(np.abs(self.iosm.e)), 0., 1.)
        self.color.r = color
        self.pub["color"].publish(self.color)
        self.pub["twist"].publish(self.T)

    def controller(self):
        # self.T.angular.x = np.random.uniform(0.0, 0.1)
        now = self.cnt_main
        print "iosm.x_raw", now, self.iosm.x_raw
        # print "now", now
        if self.cfg.mode == "vel":
            if True: # self.cnt_main % 1 == 0:
                # target_change_period = 300
                # target_change_period = 100.
                # target_change_period = 50.
                # target_change_period = 20.
                target_change_period = self.target_change_period

                # constant target
                if self.cfg.tp_target_spec["constant"]:
                    if self.cnt_main == 10:
                        # self.phi = np.random.uniform(0., np.pi/2.)
                        self.phi = np.random.uniform(0., 2 * np.pi)
                        self.vel = 0.8 # np.random.uniform(0.8, 1.2)
                    self.cfg.target[0,0] = self.vel * np.cos(self.phi)
                    self.cfg.target[1,0] = self.vel * np.sin(self.phi)
                
                # jumping target                    
                elif self.cfg.tp_target_spec["jumping"]:
                    # if self.cnt_main in range(self.cfg.len_washout+1, self.cfg.len_episode, 21):
                    # if self.cnt_main in self.target_jumps:
                    if self.cnt_main in range(self.cfg.len_washout+1, self.cfg.len_episode, int(target_change_period)):
                        # self.phi = np.random.uniform(0., np.pi/2.)
                        # self.phi = np.random.uniform(0., 2 * np.pi)
                        
                        # self.phi += np.random.uniform(-0.05, 0.05)
                        # self.phi = np.clip(self.phi, 0., 0.9 * np.pi/2)
                        self.phi += np.random.normal(0., 0.1)
                        self.phi = np.clip(self.phi, -np.pi, np.pi)
                    
                        # phi = np.random.uniform(0, np.pi)
                        # self.vel = np.random.uniform(0.8, 1.2)
                        self.vel += np.random.normal(0., 0.02)
                        self.vel = np.clip(self.vel, 0.5, 1.5)
                        # self.vel = 0.8
                        self.cfg.target[0,0] = self.vel * np.cos(self.phi)
                        self.cfg.target[1,0] = self.vel * np.sin(self.phi)
                    
                elif self.cfg.tp_target_spec["jumping_sign"]:
                    # jumping target inverting sign
                    if self.cnt_main in range(self.cfg.len_washout+1, self.cfg.len_episode, int(target_change_period)):
                        self.phi = np.random.uniform(0., 2 * np.pi)
                        self.vel = np.random.uniform(0.8, 1.2)
                        self.cfg.target[0,0] = self.vel * np.cos(self.phi)
                        self.cfg.target[1,0] = self.vel * np.sin(self.phi)
                        # if self.cnt_main % 2 == 0:
                        #     print "even"
                        #     self.cfg.target[0,0] = 1. + np.random.uniform(-0.3, 0.3)
                        #     self.cfg.target[1,0] = 1. + np.random.uniform(-0.3, 0.3)
                        # else:
                        #     print "odd"
                        #     self.cfg.target[0,0] = -1. + np.random.uniform(-0.3, 0.3)
                        #     self.cfg.target[1,0] = -1. + np.random.uniform(-0.3, 0.3)

                elif self.cfg.tp_target_spec["sine"]:
                    # # sinewave target fixed amp
                    # self.cfg.target[0,0] = 1. + 0.5 * np.cos(self.cnt_main/20.)
                    # self.cfg.target[0,0] = 0.8 * np.sin(self.cnt_main/100.)
                    # phi = (self.cnt_main % 314) / 100.
                    # phi = (self.cnt_main % 80) / 100.

                    # modulate angle continuously
                    # if self.phi >= (0.9 * np.pi/2.) or self.phi <= 0.1:
                    #     self.phi_inc = -self.phi_inc
                    # self.phi = self.phi + self.phi_inc
                    self.phi = self.cnt_main/target_change_period
                    print "self.phi", self.phi
                    # vel = 0.8 + 0.3 * np.sin(self.cnt_main / 20.)
                    # self.vel += np.random.normal(0., 0.01)
                    # self.vel = np.clip(self.vel, 0.8, 1.2)
                    self.vel = 1.0
                    print "phi", self.phi, "vel", self.vel
                    self.cfg.target[0,0] = self.vel * np.cos(self.phi)
                    self.cfg.target[1,0] = self.vel * np.sin(self.phi)
                
                    
                    
                # # sinewave with increasing amplitude
                # self.cfg.target[0,0] = 1.
                # sinamp = np.clip(self.cnt_main/4000., -0.2, 0.2)
                # self.cfg.target[0,0] += sinamp * np.sin(self.cnt_main/50.)

                print "target", self.cfg.target
                self.pub["target"].publish(self.cfg.target)
                # err = (self.iosm.x_raw[0:self.cfg.idim:2,0] - self.cfg.target.T).T # 
                err = (self.iosm.x_raw[0:2,0] - self.cfg.target.T).T # 
                # err = np.atleast_2d(self.iosm.x[0:self.cfg.idim:2,0]).T
                # err1 = -np.sum(np.square(err))
                err1 = -np.square(err)
                err2 = -np.sum(np.square(err))
                print "err", err, err1
                # self.err[self.cnt_main,0] = err1
                self.err[self.cnt_main] = err1.flatten()
                self.iosm.e = err1
                # self.iosm.e = err2 * np.ones_like(err)
                # self.iosm.t[0,0] = self.cfg.target[0,0]
                self.iosm.t[0:self.cfg.idim:2] = self.cfg.target
                if self.cnt_main > 200:
                    # print "mse window", self.err[(self.cnt_main-200):self.cnt_main,0]
                    self.iosm.mse = np.mean(np.sqrt(-self.err[(self.cnt_main-200):self.cnt_main,0])) * np.ones_like(self.cfg.target)
                    print "mse", self.iosm.mse
                else:
                    self.iosm.mse = np.ones_like(self.cfg.target)
                
                # self.err[self.cnt_main,1] = err1
                # print "self.rew.perf.shape", self.rew.perf.shape
                # print self.err[self.cnt_main,0]
                # FIXED: acceleration based reward
                # FIXME: sign is inverted when running on different machines?
                # FIXME: this performance measure asks for infinitely large accelerations
                # self.rew.perf = -np.sign(err) * np.array([self.iosm.x_raw[2,0]]).reshape((1,1)) # eta
                self.rew.perf = err1 # np.sign(err) * np.array([self.iosm.x_raw[2,0]]).reshape((1,1)) # rudi
                # self.rew.perf = err2 * np.ones_like(err)
                # self.rew.perf = (err2 + self.iosm.e) * np.ones_like(err)
                # self.rew.perf[1,0] = err1
                
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
                # clamp to zero
                # self.iosm.x[1] = 0.
                # self.iosm.x[3] = 0.
                self.iosm.z = self.res.execute(self.iosm.x)
                self.iosm.zn = self.res.zn
                # print "res.theta", self.res.theta
                # print "z/zn", self.iosm.z, self.iosm.zn
                # print "z/zn.shape", self.iosm.z.shape, self.iosm.zn.shape
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
            print "self.err", self.err
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
            print "bump"
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
            print "ramp"
            self.iosm.z[0,0] = (250 - np.abs((self.cnt_main % 500) - 250)) * 0.005
            if np.random.uniform(0, 1.) > 0.9:
                print "hickup"
                time.sleep(0.5)

        # store states for pushback: see above

        print "publishing zn"
        self.pub["learn_zn"].publish(self.iosm.zn)
        return(self.iosm.z, self.iosm.zn)


def main(args):
    # fix RNG seed
    np.random.seed(args.seed)

    # check datadir exists
    if not os.path.exists(args.datadir):
        print "Datadir doesn't exist, try mkdir '%s'" % (args.datadir)
    
    if args.mode == "res_gen":
        # pre-generate N reservoirs for use in experiments
        for i in range(10):
            s = SpheroResLearner2D(args)
            timestamp = time.strftime("%Y-%m-%d-%H%M%S")
            filename = "%s/reservoir-%d-%s.bin" % (args.datadir, i, timestamp)
            s.res.save(filename=filename)
        s.isrunning = False
        sys.exit(0)
    else:
        s = SpheroResLearner2D(args)
        # do we get passed a trained network for testing?
        if args.resfile != "":
            # then load it
            print "loading reservoir from file %s" % (args.resfile)
            s.res.load(filename="%s/%s" % (args.datadir, args.resfile))
            # bend some parameters, e.g. eta and theta
            print s.res.theta
            s.cfg.eta_EH = 0.
            s.cfg.theta = 1e-5
            s.res.set_theta(s.cfg.theta)
            print "overriding config for eta, theta", s.cfg.eta_EH, s.res.theta
   
    def handler(signum, frame):
        print 'Signal handler called with signal', signum
        s.isrunning = False
        sys.exit(0)
        # raise IOError("Couldn't open device!")
    
    signal.signal(signal.SIGINT, handler)
    
    s.start()
    while not rospy.is_shutdown():
        # print("main loop")
        time.sleep(1)

    # timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # save logs
    target_str = s.cfg.tp_target_spec.keys()[s.cfg.tp_target_spec.values().index(1)]
    print "target_str", target_str
    filename = "%s/log-learner-%s-N-%d-eta-%f-theta-%f-g-%f-target-%s" % (args.datadir,
                                                                                   timestamp,
                                           s.cfg.N, s.cfg.eta_EH, s.cfg.res_theta, s.cfg.g,
                                           target_str)
    s.savelogs(ts=timestamp, filename=filename)
    # save network
    if args.resfile == "":
        filename = "%s/reservoir-%d-%s-N-%d-eta-%f-theta-%f-g-%f-target-%s.bin" % (args.datadir,
                                                                                   1, timestamp,
                                           s.cfg.N, s.cfg.eta_EH, s.cfg.res_theta, s.cfg.g,
                                           target_str)
    #    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    #    filename = "%s/reservoir-%d-%s.bin" % (args.datadir, 1, timestamp)
    s.res.save(filename=filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reservoir smp learner for sphero")
    parser.add_argument("-c", "--config", dest="cfgfilename", help="config file to use",
                        default="default.cfg")
    parser.add_argument("-d", "--datadir", help="Directory root for saved configurations",
                        default=parser.prog[0:-3])
    parser.add_argument("-m", "--mode", type=str, help="select mode: " + str(SpheroResLearner2D.modes),
                        default="vel")
    parser.add_argument("-rf", "--resfile", dest="resfile", help="pickled reservoir to load",
                        default="")
    parser.add_argument("-s", "--seed", dest="seed", help="seed for rng", default=123, type=int)
    args = parser.parse_args()

    # good seeds: 4, 
    
    main(args)
