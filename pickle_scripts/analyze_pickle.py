import pickle as pickle
import warnings
import time, argparse, sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import datasets, linear_model
from sklearn import kernel_ridge


pickleFolder = '../goodPickles/'

class Analyzer():
    def __init__(self,args):
        self.args = args

        if args.randomFile:
            files = [f for f in self.all_files(pickleFolder)
                if f.endswith('.pickle') and 'recording' in f]
            self.filename = files[np.random.randint(len(files))]
            print("random File: %s" % (self.filename))
        else:
            self.filename = self.args.filename

        try:
            self.variableDict = pickle.load(open( self.filename, "rb" ) )
        except Exception:
            raise Exception("File not found, use -f and the filepath")

        if self.variableDict["dataversion"] == 1:
            warnings.warn("Pickle from V1, xsi might not be correct")
        self.nummot = self.variableDict["nummot"]
        self.motorCommands = self.variableDict["y"][:,:,0]
        self.sensorValues = self.variableDict["x"][:,:,0]
        self.timesteps = len(self.motorCommands)

        self.windowsize = self.args.windowsize
        self.embsize = self.args.embsize
        self.hamming = self.args.hamming

    def all_files(self,directory):
        for path, dirs, files in os.walk(directory):
            for f in files:
                yield os.path.join(path, f)

    def get_triu_of_matrix(self,matrix):
        if matrix.shape[0] != matrix.shape[1]: return None

        dim = matrix.shape[0]
        triu = np.triu_indices(dim, k=1)
        return matrix[triu]

    def correlation_func(self):
        loopsteps = self.timesteps - self.windowsize

        self.motorCommands += np.random.normal(size=self.motorCommands.shape) * 0.1

        # testdata
        # self.motorCommands = np.arange(20).reshape(10,2)
        # print self.motorCommands

        # print(len(self.motorCommands))


        allCorrelations = np.zeros((loopsteps, self.nummot, self.nummot))
        allAverageSquaredCorrelations = np.zeros((loopsteps, 6))


        correlationsGlobal = np.cov(self.motorCommands.T)
        print(correlationsGlobal)

        for i in range(len(self.motorCommands) - self.windowsize):
            # use only the latest step of the buffer but all motor commands
            # shape = (timestep, motorChannel, buffer)

            window = self.motorCommands[i:i+self.windowsize,:]
            #print "windooriginal", window

            windowfunction= np.hamming(self.windowsize)
            if(self.hamming):
                window = (window.T * windowfunction).T
            #print "windowshapehamming", window

            correlations = np.cov(window.T)

            # normalize
            for x in range(4):
                for j in range(4):
                    correlations[x,j] = correlations[x,j] / np.sqrt(correlationsGlobal[x,x] * correlationsGlobal[j,j])

            if self.hamming:
                correlations[:,:] *= self.windowsize / np.sum(windowfunction)
            allCorrelations[i,:,:] = correlations[:,:]



            # save average of the squared upper triangle of the correlations

            # allAverageSquaredCorrelations[i,:] = np.triu(correlations,k=1).flatten() ** 2
            #allAverageSquaredCorrelations[i,:] = np.triu(correlations,k=1).flatten()

            #allAverageSquaredCorrelations[i,0] = np.sum(np.triu(correlations,k=1).flatten() ** 2)
            #allAverageSquaredCorrelations[i,:] = np.abs(np.triu(correlations,k=1).flatten())
            allAverageSquaredCorrelations[i,:] = self.get_triu_of_matrix(correlations)


        corrCombinations = allAverageSquaredCorrelations.shape[1]
        #print "corrCombinations", allAverageSquaredCorrelations[0,:]

        combinationNames = ["rb-lb", "rb-rf", "rb-lf", "lb-rf", "lb-lf", "rf-lf"]
        numtimesteps = allAverageSquaredCorrelations.shape[0]

        colors = cm.jet(np.linspace(0, 1, corrCombinations))
        colors = ["#FFFF00", "#FF0000", "#FF00FF", "#0000FF", "#00FFFF", "#00FF00"]

        fig = plt.figure(figsize=(15,5))

        xValues = (np.arange(len(allAverageSquaredCorrelations[:,j])) + (self.windowsize // 2)) / 20.

        plt.plot(xValues, [0] * len(xValues), 'k', alpha=0.3)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        print(np.average(allAverageSquaredCorrelations[:,0]))


        for j in range(corrCombinations):
            plt.plot(xValues, allAverageSquaredCorrelations[:,j], alpha = 1, c = colors[j], label=combinationNames[j])

        #plt.plot(xValues, np.average(np.abs(allAverageSquaredCorrelations[:,:]), axis=1), c='k', lw = 2, label="average")
        #plt.legend()


        t= 0
        tstart = 0
        printingChannel = -1
        while t < numtimesteps:
            maxChannel = np.argmax(allAverageSquaredCorrelations[t,:])
            if maxChannel != printingChannel or t == numtimesteps - 1:
                if printingChannel != -1:
                    # plot old channel
                    if tstart < 0:
                        tstart = 0
                    plt.plot(np.arange(tstart + 50, t + 50), allAverageSquaredCorrelations[tstart:t ,printingChannel], c = colors[printingChannel], lw=3)

                # change to new printingChannel
                printingChannel = maxChannel
                tstart = t - 1


            t+=1

        #plt.plot(allAverageSquaredCorrelations[:,0])
        #plt.title("Correlation from a episode with")
        plt.xlabel("time [s]", fontsize=28)
        plt.ylabel("correlation", fontsize= 28)
        #plt.ylim((-0.5,0.5))
        plt.xlim(0,self.timesteps / 20)
        plt.xticks(np.linspace(0,50,11))
        plt.tight_layout()
        fig.savefig('correlations.png')
        plt.show()
        #print allAverageSquaredCorrelations

    def prepare_data_for_learning(self):
        testDataLimit = 4 * self.timesteps / 5

        motoremb = np.array([self.motorCommands[i:i+self.embsize].flatten() for i in range(0, testDataLimit - self.embsize)])
        motorembtest = np.array([self.motorCommands[i:i+self.embsize].flatten() for i in range(testDataLimit, self.timesteps - self.embsize)])
        #self.sensorValues = self.sensorValues[:,3:]

        # motoremb = self.motorCommands[:testDataLimit]
        print((motoremb.shape))
        self.trainingData={"motor":motoremb, "sensor":self.sensorValues[self.embsize:testDataLimit]}
        self.testData={"motor":motorembtest, "sensor":self.sensorValues[testDataLimit+self.embsize:]}


    def learn_motor_sensor_gmm(self):
        """
        This mode learns the sensor responses from the motor commands with
        gaussian mixture models and tests the result
        """
        # TODO: implement

        return 0


    def learn_motor_sensor_linear(self):
        """
        This mode learns the sensor responses from the motor commands with
        linear regression and tests the result
        """

        self.prepare_data_for_learning()

        # regr = linear_model.LinearRegression()
        regr = linear_model.Ridge(alpha = 10.0)
        #regr = kernel_ridge.KernelRidge(alpha = 0.5, kernel="rbf")
        regr.fit(self.trainingData["motor"], self.trainingData["sensor"])

        # predict the sensor data from the motor data
        predTest = regr.predict(self.testData["motor"])

        # find the absolute sum of the coeffitients corresponding to the lag coefs[max] is zero lag
        coefs = np.sum(np.sum(np.abs(regr.coef_), axis = 0).reshape((4, self.embsize)), axis = 0)
        print(coefs)
        print(np.argmax(coefs))

        # calculate mean squared error of the test data
        mse = np.mean((predTest - self.testData["sensor"]) ** 2)

        #print("trainingError: %.2f" %np.mean((regr.predict(trainingData["motor"]) - trainingData["sensor"]) ** 2))
        print(("Mean squared error: %.2f, s = %s" % (mse, np.var(self.testData["sensor"], axis = 0))))

        # plot the coefficient sum
        plt.plot(coefs, label="coefefs over lag")

        fig, axs = plt.subplots(nrows=3, ncols=4)

        bins = np.linspace(-2, 2, 15)
        for j, ax in enumerate(axs.reshape(-1)):
            i = j / 2

            predError = predTest[:,i] - self.testData["sensor"][:,i]

            # get the sensor values without mean
            zmData = self.testData["sensor"][:,i] - np.mean(self.testData["sensor"][:,i])

            # plot time series
            if j % 2 == 0:
                ax.plot(self.testData["sensor"][:,i])
                ax.plot(predTest[:,i])
            else:
                """
                plot a histogram of the prediction error and the zero mean
                data, the prediction error should be sharper than the signal
                distribution
                """
                data = np.vstack([predError, zmData]).T
                ax.hist(data, bins=bins, alpha=1, label=["predictionError", "meanedData"])
                ax.legend()

        plt.show()


    def scattermatrix_func(self):
        """
        This function creates a scatterplot of the data
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        from pandas.tools.plotting import scatter_matrix
        import matplotlib.cm as cm

        # settings, ammount of points in one graph and color mode
        points_per_plot = 501 # self.timesteps
        color = False

        if color:
            c = np.linspace(0,1,points)
        else:
            c = None

        # go through the data until the end
        plots = int(self.timesteps / points_per_plot)
        print("Warning: %d points not shown" % (self.timesteps % points_per_plot))

        for part in range(0,plots):
            partS = part * points_per_plot
            partE = (part + 1) * points_per_plot
            combinedData =  np.hstack((self.motorCommands[partS:partE,:], self.sensorValues[partS:partE,:]))

            df = pd.DataFrame(combinedData, columns=['m1', 'm2', 'm3', 'm4', 's1', 's2', 's3', 's4', 's5', 's6'])


            scatterm = scatter_matrix(df, alpha=0.5, s=25, figsize=(12, 12), diagonal='kde', c= c, edgecolors='none')

            for x in range(10):
                for y in range(10):
                    scatterm[x,y].set_ylim(-1.5,1.5)
                    scatterm[x,y].set_xlim(-1.5,1.5)

            #plt.draw()
            #plt.show()
            #plt.savefig("scattermatrix%d.jpg" %(part))
            #print("scatter %d saved" %(part))
            plt.show()

    def spectogram_func(self):
        loopsteps = self.timesteps - self.windowsize

        sensors = self.sensorValues[...]
        self.numsen = sensors.shape[1]
        s = sensors

        print(sensors.shape)

        import scipy.signal as sig

        m = self.motorCommands
        s = self.sensorValues

        Mspecs = [sig.spectrogram(m[:,i], fs=20.0, nperseg = 32, nfft=32) for i in range(self.nummot)]
        Sspecs = [sig.spectrogram(s[:,i], fs=20.0, nperseg = 32, nfft=32) for i in range(self.numsen)]

        allSpecs = np.zeros((35,0))

        from matplotlib import gridspec
        gs = gridspec.GridSpec(3,max(self.nummot, self.numsen)+1)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[0,0])
        ax1.plot(sensors)
        for i in range(self.numsen):
            ax = fig.add_subplot(gs[0,1+i])

            Mspec = Sspecs[i]
            ax.pcolormesh(Mspec[1], Mspec[0], Mspec[2])

            allSpecs = np.concatenate((allSpecs, Mspec[2].T), axis= 1)

        ax2 = fig.add_subplot(gs[1,0])
        ax2.plot(m)
        for i in range(self.nummot):
            ax = fig.add_subplot(gs[1,1+i])

            Mspec = Mspecs[i]
            ax.pcolormesh(Mspec[1], Mspec[0], Mspec[2])

            allSpecs = np.concatenate((allSpecs, Mspec[2].T), axis = 1)

        # do k-means
        from sklearn.cluster import KMeans
        import matplotlib.cm

        kmeans = KMeans(n_clusters=4, random_state=1)
        kmeans.fit(allSpecs)
        ax3 = fig.add_subplot(gs[2,1])

        ax3.scatter(list(range(len(allSpecs))), kmeans.predict(allSpecs),c=np.linspace(0,255,35))

        from sklearn import decomposition

        pca = decomposition.PCA(n_components=2)
        pca.fit(allSpecs)
        ax4 = fig.add_subplot(gs[2,2])
        ax4.scatter( pca.transform(allSpecs)[:,0], pca.transform(allSpecs)[:,1], c=np.linspace(0,255,35))

        print((allSpecs.shape))
        plt.show()

        print(sensors.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lpzrobots ROS controller: test homeostatic/kinetic learning")
    parser.add_argument("-m", "--mode", type=str, help="cor = correlation, scat = scattermatrix, spect = spectogram, lin = linear regression learning")
    parser.add_argument("-f", "--filename", type=str, help="filename (no default)", nargs='?')
    parser.add_argument("-r", "--randomFile", action="store_true")
    parser.add_argument("-ham", "--hamming", action="store_true")
    parser.add_argument("-w", "--windowsize", type=int, help="correlation window size", default=100)
    parser.add_argument("-es", "--embsize", type=int, help="history time steps for learning", default=15)
    args = parser.parse_args()

    function_dict = {
    'cor': Analyzer.correlation_func,
    'scat':Analyzer.scattermatrix_func,
    'spect':Analyzer.spectogram_func,
    'lin': Analyzer.learn_motor_sensor_linear
    }

    if(args.filename == None and args.randomFile == False):
        print("Please select file with\n-f ../foo.pickle or -r for random file\n")

    if(args.mode in list(function_dict.keys())):
        print("Mode %s selected...\n" % (args.mode))

        analyzer = Analyzer(args)
        function_dict[args.mode](analyzer)
    else:
        if args.mode == None:
            args.mode=""

        print(("Mode '" + args.mode + "' not found,\nplease select a mode with -m " + str(list(function_dict.keys()))))
