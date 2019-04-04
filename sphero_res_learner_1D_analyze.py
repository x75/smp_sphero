#!/usr/bin/env python
__author__ = "Oswald Berthold <bertolos@informatik.hu-berlin.de>"
"""Analyze results from sphero 1D and 2D learning experiments"""

import numpy as np
import pandas as pd
import matplotlib.pylab as pl

import argparse, sys

pl.rcParams["pdf.fonttype"] = 42
pl.rcParams["ps.fonttype"] = 42

class SpheroResLearner1DAnalyze(object):
    def __init__(self, args):
        self.name = "SpheroResLearner1DAnalyze"
        self.logfiles_num = len(args.logfiles[0])
        self.logfiles = args.logfiles[0]
        self.saveplot = args.saveplot
        self.X = []

    def load(self):
        try:
            for logfile in self.logfiles:
                self.X.append(np.load(logfile))
        except Exception as e:
            print(e)

    def plot_x_raw(self):
        for i, logdata in enumerate(self.X):
            pl.subplot(211)
            pl.title("vel x")
            pl.plot(logdata["x_raw"][:,0])
            # pl.plot(self.X["x_raw"][:,0], self.X["x_raw"][:,1])
            # pl.gca().set_aspect(1)
            pl.subplot(212)
            pl.title("err x")
            pl.plot(logdata["e"][:,0])
        pl.show()

    def plot_x(self):
        for i, logdata in enumerate(self.X):
            pl.subplot(211)
            pl.title("vel x")
            pl.plot(logdata["x"][:,0])
            # pl.plot(logdata["x_raw"][:,0], logdata["x_raw"][:,1])
            # pl.gca().set_aspect(1)
            pl.subplot(212)
            pl.title("err x")
            pl.plot(logdata["x"][:,1])
        pl.show()

    def plot_all(self):
        # microscope ref
        micro_x = [80, 120]
        for i, logdata in enumerate(self.X):
            # sl = slice(None)
            sl = slice(0, 2000)
            pl.subplot(511)
            pl.title("x_raw")
            pl.plot(logdata["x_raw"][sl])
            pl.gca().set_xticklabels([])

            pl.subplot(512)
            pl.title("x")
            pl.plot(logdata["x"][sl])
            try:
                pl.plot(logdata["t"][sl], "-o")
            except Exception as e:
                print("fail", e)

            pl.gca().set_xticklabels([])
                
            pl.subplot(513)
            pl.title("z")
            pl.plot(logdata["z"][sl])
            pl.plot(logdata["zn"][sl])
            pl.plot(logdata["zn_lp"][sl])
            pl.gca().set_xticklabels([])

            pl.subplot(514)
            pl.title("err")
            try:
                pl.plot(np.abs(logdata["e"][sl]))
            except Exception as e:
                print("fail", e)
            pl.gca().set_yscale("log")
            # pl.gca().set_xscale("log")
            pl.gca().set_xticklabels([])
            
            pl.subplot(515)
            pl.title("w")
            w_ = np.linalg.norm(logdata["w"], axis=1)[sl]
            print("w_.shape", w_.shape)
            dw = np.diff(logdata["w"], axis=0)
            print("dw.shape", dw.shape)
            dw_ = np.linalg.norm(dw[sl], axis=1)
            pl.plot(dw_, "-o")
            pl.xlabel("t [steps]")

        # mark icroscoped region
        for j in range(5):
            pl.subplot(5, 1, j+1)
            pl.axvspan(micro_x[0], micro_x[1], 0., 1., facecolor="k", alpha=0.2)
            
        if self.saveplot:
            pl.gcf().set_size_inches((12,7))
            pl.gcf().savefig("%s-plot_all.pdf" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
        pl.show()

    def plot_microscope(self):
        for i, logdata in enumerate(self.X):
            # sphero_res_learner_1D/log-learner-20150315-233141-eta-0.001000-theta-0.200000-g-0.999000-target-sine.npz
            # sphero_res_learner_1D/log-learner-20150315-230854-eta-0.001000-theta-0.200000-g-0.999000-target-sine.npz
            # sl = slice(70, 140)
            sl = slice(80, 120)
            # sl = slice(0, 1000)
            # spikes = [19, 27, 30, 32]
            spikes = [20,22,24]
            # pl.subplot(511)
            # pl.title("x_raw")
            # pl.plot(logdata["x_raw"][sl])
            pl.subplot(111)
            pl.title("Motor signal, performance and weight changes")
            # x = logdata["x"][sl,0]
            x = logdata["e"][sl,0]
            # dx = np.diff(x, axis=0)
            dx = x + np.roll(x, -1)
            # dx = np.roll(dx, 1)
            # dx[0] = 0
            # for j in spikes:
            #     pl.axvline(x=j+3, ymin=0.45, ymax=0.55, lw=2)
            # x = np.roll(x, -3)
            # recreate perf_lp
            dx_lp = np.zeros_like(dx)
            for j,k in enumerate(dx):
                if j > 1:
                    dx_lp[j] = 0.8 * dx_lp[j-1] + 0.2 * k
                else:
                    dx_lp[j] = k
            pl.plot(dx    * 0.5, lw=2, label="perf")
            # pl.plot(x, lw=2, label="P")
            pl.plot(dx_lp * 0.5, lw=2, label="perf_lp")
            # try:
            #     pl.plot(logdata["t"][sl,0])
            # except Exception, e:
            #     print "fail", e
            # pl.subplot(111)
            # pl.title("z")
            pl.plot(logdata["z"][sl], lw=2, label="Motor")
            zn = logdata["zn"][sl]
            pl.plot(zn, lw=2, label="Noisy motor")
            # pl.plot(logdata["zn_lp"][sl])
            # for j in spikes:
            #     pl.axvline(j, 0.70, 0.85, color="k", linestyle="--", lw=2)
            print(x.shape, zn.shape)
            xzcorr = np.correlate(zn[:,0], x, "full")
            print("x-z corr", xzcorr)
            # pl.plot(xzcorr)
            # pl.subplot(514)
            # pl.title("err")
            # try:
            #     pl.plot(np.abs(logdata["e"][sl]))
            # except Exception, e:
            #     print "fail", e
            # pl.gca().set_yscale("log")
            # pl.gca().set_xscale("log")
            # pl.subplot(111)
            # pl.title("w")
            w_ = np.linalg.norm(logdata["w"], axis=1)[sl]
            print("w_.shape", w_.shape)
            dw = np.diff(logdata["w"], axis=0)
            print("dw.shape", dw.shape)
            dw_ = np.linalg.norm(dw[sl], axis=1)
            # dw_ = np.roll(dw_, -3)
            pl.plot(dw_ * 100. - 4., "-o", lw=2, label="|dW|")
            for j in spikes:
                # pl.axvline(j, 0.70, 0.85, color="k", linestyle="--", lw=2)
                # pl.axvline(j+3, 0.05, 0.55, color="k", linestyle="--", lw=2)
                pl.plot([j, j], [zn[j]-0.5, zn[j]+0.5], color="k", linestyle="--", lw=2)
                pl.plot([j, j+3], [zn[j]-0.5, x[j+3]+0.3], color="k", linestyle="--", lw=2)
                pl.plot([j+3, j+3], [x[j+3]+0.3, -4.], color="k", linestyle="--", lw=2)
                pl.axvline(j+3, 0.05, 0.55, color="k", linestyle="--", lw=2)
            zwcorr = np.correlate(zn[:,0], dw_[:,0], "full")
            print("z-w corr", zwcorr)
            pl.text(16, -0.5, "$k$-delay")
            # pl.plot(zwcorr)
            # pl.subplot(511)
            # pl.title("corrs")
            # pl.plot(xzcorr, "k.", label="x/z")
            # pl.twinx()
            # pl.plot(zwcorr, ".", label="z/w")
            pl.ylim((-4.5, 2.))
            pl.ylabel("Mixed")
            pl.xlabel("t [steps]")

            ax = pl.gca()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            
        if self.saveplot:
            pl.gcf().set_size_inches((12,7))
            pl.gcf().savefig("%s.pdf" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
        
        pl.show()
       
                
    def plot_z(self):
        for i, logdata in enumerate(self.X):
            pl.plot(logdata["z"])
        pl.show()
        
    def plot_w(self):
        for i, logdata in enumerate(self.X):
            w = logdata["w"]
            print(w.shape)
            w_ = np.sqrt(np.sum(np.square(w), axis=1))
            print(w_.shape)
            pl.plot(w_)
        pl.show()

    def plot_bump(self):
        dim = self.X["x_raw"].shape[1]
        pl.subplot(111)
        pl.title("x")
        pl.plot(self.X["z"][:,0], "-ko")
        pl.plot(self.X["x_raw"][:,0], "-bo")
        pl.show()

    def plot_xcorr(self):
        print(self.X["x_raw"].shape)
        print(self.X["z"].shape)
        # tsl = slice(40, 290)
        tsl = slice(None, None)
        x_raw = self.X["x_raw"][tsl]
        z = self.X["z"][tsl]
        print(x_raw[:,0])
        xcorr = np.correlate(z[:,0], x_raw[:,0], mode="full")
        pl.subplot(211)
        # pl.plot(np.roll(x_raw[:,0], -3))
        pl.plot(x_raw[:,0])
        pl.plot(z)
        pl.subplot(212)
        pl.text(100, 200, "len(xcorr) = %d, len(z) = %d" % (len(xcorr), len(z)))
        pl.text(100, 140, "argmax(xcorr) = %d" % np.argmax(xcorr))
        pl.axvline(np.argmax(xcorr))
        pl.plot(xcorr)
        pl.show()

    def plot_err(self):
        err = np.zeros_like(self.X[0]["e"])
        print(err)
        for i, logdata in enumerate(self.X):
            err += logdata["e"]
        pl.subplot(111)
        # pl.title("error")
        pl.plot(-err/self.logfiles_num, "k.", lw=0.5, ms=1.)
        # pl.plot(-err)
        # pl.gca().set_xscale("log")
        pl.ylim((1e-5, 1e1))
        pl.gca().set_yscale("log")
        pl.xlabel("time step")
        pl.ylabel("log MSE")
        # print sys.argv[0]
        if self.saveplot:
            pl.gcf().set_size_inches((5,2))
            # pl.gcf().savefig("%s.pdf" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
            pl.gcf().savefig("%s.jpg" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
        pl.show()

    def plot_err_episodes(self):
        washout = 10
        err = np.zeros((49, 1))
        print(err)
        for i, logdata in enumerate(self.X):
            for j in range(49):
                endofepi = logdata["e"][j*101+50:((j+1)*101)]
                print(endofepi)
                err[j,0] += np.mean(endofepi)
        pl.subplot(111)
        # pl.title("error")
        # pl.plot(-err/self.logfiles_num, "k.", lw=0.5)
        pl.plot(-err, "ko", lw=0.5)
        # pl.plot(-err)
        # pl.gca().set_xscale("log")
        pl.gca().set_yscale("log")
        pl.xlim((-1, None))
        pl.xlabel("time step / 100")
        pl.ylabel("log error")
        # print sys.argv[0]
        if self.saveplot:
            pl.gcf().set_size_inches((9,3))
            pl.gcf().savefig("%s.pdf" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
        pl.show()

    def plot_test_jumping(self):
        # target_change_period = 62
        ldim = self.X[0]["z"].shape[1]
        if ldim == 1:
            target_change_period = 101 # 1D
            target_change_onset = 11
        elif ldim == 2:
            target_change_period = 100 # 2D
            target_change_onset = 12
        err = [np.zeros((target_change_period, ldim)) for i in range(len(self.X))]
        print("len(X)", len(self.X))
        for i, logdata in enumerate(self.X):
            for j in range(target_change_onset,
                           len(logdata["e"])-(target_change_period-target_change_onset),
                           target_change_period):
                print("j", j)
                lerr = logdata["e"][j:(j+target_change_period)]
                # pl.plot(lerr)
                # pl.show()
                # blub = lerr
                # print "blub", blub.shape, blub
                err[i] += -lerr
                # pl.plot(blub)
        # print len(err)
        cols = ["k", "r"]
        # cols = ["r", "b"]
        labs = ["ctlr 1", "ctlr 2"]
        pl.subplot(211)
        # sl = slice(10, 1010)
        if ldim == 1:
            sls = 50
            sle = 550
        elif ldim == 2:
            sls = 450
            sle = 950
        sl = slice(sls, sle)
        t = list(range(sls, sle))
        # for 1D
        if ldim == 1:
            pl.plot(t, self.X[i]["t"][sl,0], "k-", lw=0.5, label="target")
        elif ldim == 2:
            # for 2d no target
            pl.plot(t, self.X[i]["t"][sl,[0]], "k-", lw=0.5, label="target")
        # pl.plot(self.X[i]["t"][10:1010,[2]], "b-", lw=0.5, label="target")
        for i in range(len(self.X)):
            # pl.plot(self.X[i]["x_raw"][10:1010,0], cols[i] + "-", lw=0.5, label="vel")
            pl.plot(t, self.X[i]["x_raw"][sl,0], cols[i] + "-", lw=2., label="ctlr %d" % (i+1))
            # pl.plot(t, self.X[i]["x_raw"][sl,1], cols[i+1] + "-", lw=1., label="vel")
            # pl.plot(self.X[i]["z"][:,0], "r-", lw=0.5, label="motor out")
        # pl.plot(self.X[0]["zn"][:,0], "g-", lw=0.5, label="motor out")
        pl.ylabel("vel [m/s]")
        if ldim == 2:
            pl.ylim((-1., 1.8))
            # pass

        # legend outside
        ax = pl.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
        # pl.legend()
        pl.subplot(212)
        pl.title("Average error on setpoint change")
        for i in range(len(self.X)):
            print("i", i, cols[i], err[i].shape)
            pl.plot(err[i][:,0], cols[i] + "-", label=labs[i], lw=2)

        # legend outside
        ax = pl.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
        # pl.legend()
        pl.xlabel("time steps")
        pl.ylabel("squared error")
        pl.gca().set_yscale("log")
        if self.saveplot:
            pl.gcf().set_size_inches((9,6))
            # make figure slightly larger to improve readability
            # pl.gcf().set_size_inches((9,7))
            pl.gcf().savefig("%s-test_jumping.pdf" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
        pl.show()

    def plot_test_transfer(self):
        # # target_change_period = 100
        # err = np.zeros((target_change_period, 1))
        # for i, logdata in enumerate(self.X):
        #     for j in range(11, len(logdata["e"])-89, target_change_period+1):
        #         blub = logdata["e"][j:(j+target_change_period)]
        #         # print "blub", blub.shape, blub
        #         err += -blub
        #         # pl.plot(blub)
        # pl.subplot(211)
        # pl.plot(self.X[0]["t"][:,0], "k-", lw=0.5, label="target")
        # pl.plot(self.X[0]["x_raw"][:,0], "b-", lw=0.5, label="vel")
        # pl.plot(self.X[0]["z"][:,0], "r-", lw=0.5, label="motor out")
        # # pl.plot(self.X[0]["zn"][:,0], "g-", lw=0.5, label="motor out")
        # pl.legend()
        # pl.subplot(212)
        # pl.title("Average error on setpoint change")
        # pl.plot(err, "k-", lw=0.5)
        # pl.xlabel("time steps")
        # pl.ylabel("squared error")
        # pl.gca().set_yscale("log")
        # if self.saveplot:
        #     pl.gcf().set_size_inches((9,5))
        #     pl.gcf().savefig("%s-test_jumping.pdf" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
        # pl.show()
        data = self.X[0]
        # print data
        pl.title("Motor transfer curve")
        pl.plot(data["zn"] * 100., data["x"][:,0], "k-", lw=0.5)
        pl.xlabel("Raw motor out [int]")
        pl.ylabel("Measured velocity [m/s]")
        if self.saveplot:
            pl.gcf().set_size_inches((9,3))
            pl.gcf().savefig("%s-test_transfer.pdf" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
        pl.show()
                
    def print_keys(self):
        for i,logdata in enumerate(self.X):
            print("X.keys()", list(logdata.keys()))

def main(args):
    a = SpheroResLearner1DAnalyze(args)
    a.load()

    if args.mode == "x_raw":
        a.plot_x_raw()
    elif args.mode == "keys":
        a.print_keys()
    elif args.mode == "x":
        a.plot_x()
    elif args.mode == "w":
        a.plot_w()
    elif args.mode == "z":
        a.plot_z()
    elif args.mode == "bump":
        a.plot_bump()
    elif args.mode == "all":
        a.plot_all()
    elif args.mode == "xcorr":
        a.plot_xcorr()
    elif args.mode == "err":
        a.plot_err()
    elif args.mode == "err_episodes":
        a.plot_err_episodes()
    elif args.mode == "test_jumping":
        a.plot_test_jumping()
    elif args.mode == "test_transfer":
        a.plot_test_transfer()
    elif args.mode == "microscope":
        a.plot_microscope()

if __name__ == "__main__":
    modes = ["x_raw", "keys", "z", "bump", "xcorr", "x", "all", "w", "err", "err_episodes", "test_jumping", "test_transfer", "microscope"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="x_raw", help="one of: " + ", ".join(modes))
    parser.add_argument("-l", "--logfiles", action="append", nargs="+",
                        default=[])
    parser.add_argument("--saveplot", dest="saveplot", action="store_true")
    parser.add_argument("--no-saveplot", dest="saveplot", action="store_false")

    args = parser.parse_args()
    print(args.logfiles)
    main(args)
