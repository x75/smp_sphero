# IMPORTANT THIS HAS A BUG IN IT ABOUT THE DIMENSIONS OF allSpecs

import time, argparse, sys, os

import pickle as pickle
import numpy as np

from sklearn import decomposition
from sklearn.cluster import KMeans
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib import gridspec, cm

def crawlFiles(args):

    pickleFolder = '../goodPickles/'
    overviewFile = 'overview.pickle'
    overviewComplete = False

    def all_files(directory):
        for path, dirs, files in os.walk(directory):
            for f in files:
                yield os.path.join(path, f)

    def saveToOverview(variableDict):
        filename = os.path.basename(variableDict["filename"])
        overviewDict[filename] = {}
        overviewDict[filename]["timesteps"] = variableDict["timesteps"]
        overviewDict[filename]["epsC"] = variableDict["epsC"]
        overviewDict[filename]["epsA"] = variableDict["epsA"]
        overviewDict[filename]["creativity"] = variableDict["creativity"]
        overviewDict[filename]["n"] = variableDict["timesteps"]
        overviewDict[filename]["dataversion"] = variableDict["dataversion"]
        if variableDict["dataversion"] >=3:
            overviewDict[filename]["lag"] = variableDict["lag"]



    files = [f for f in all_files(pickleFolder)
        if f.endswith('.pickle') and 'recording' in f]

    if os.path.exists(pickleFolder + overviewFile):
        overviewDict = pickle.load(open(pickleFolder + overviewFile, "rb"))
        print("overviewDict found")
        if(len(overviewDict) == len(files)):
            print("overviewDict is complete")
            overviewComplete = True
        else:
            print("overviewDict is not complete rebuilding")
            overviewDict = {}
        #print overviewDict
    else:
        overviewDict = {}

    epsArray = np.zeros((len(files), 2))
    dominantFrequencies = np.zeros((len(files)))
    i = j = 0

    numtimesteps = 1000
    nperseg = 128
    noverlap = nperseg // 8
    frames_per_seg = (int)(np.floor(numtimesteps / (nperseg - noverlap)))
    frequency_count = nperseg // 2 + 1 # 32nperseg -> 17 256nperseg -> 129
    frequency_of_all_modularities = frequency_count * 10
    y = np.zeros((0,4))
    x = np.zeros((0,6))
    c = np.zeros((0))
    ckl = []

    MspecDict = {}
    SspecDict = {}
    allSpecs = np.zeros((frequency_of_all_modularities,0))
    allSpecsSum = np.zeros((frequency_count, 1))
    print("allSpecsSumshape", allSpecsSum.shape)

    def concatenate_all_chanels(Mspecs, Sspecs):

        # get the size of the spectogram of the first motor channel (number of frames)
        specs = np.zeros((0,frames_per_seg))
        for Mspec in Mspecs:
            specs = np.concatenate((specs, Mspec[2]))
        for Sspec in Sspecs:
            specs = np.concatenate((specs, Sspec[2]))
        return specs

    def sum_all_channels_and_frames(Mspecs, Sspecs):
        specsSum = np.zeros((frequency_count, 1))
        for Mspec in Mspecs:
            specsSum = specsSum +  np.sum(np.matrix(Mspec[2]), axis=1)
        for Sspec in Sspecs:
            specsSum = specsSum + np.sum(np.matrix(Sspec[2]), axis=1)

        print("specssumshape" , specsSum.shape)
        return specsSum

    # Check if file is good
    def check_variable_dict(variableDict):
        if variableDict["timesteps"] != numtimesteps: return False
        if variableDict["dataversion"] <= 3: return False
        if variableDict["lag"] != 2: return False
        if "weight_in_body" in variableDict: return False
        return True

    print("This is gonna take some time")
    for f in files:
        filename = os.path.basename(f)
        if(overviewComplete):

            if not(check_variable_dict(overviewDict[filename])):
                print("file skiped")
                continue

        print(("Unpickling %d of %d" % ( i, len(files))))
        variableDict = pickle.load(open( f, "rb" ) )

        # fix for files without filename
        if not "filename" in variableDict:
            variableDict["filename"] = f

        saveToOverview(variableDict)

        if(not "dataversion" in variableDict):
            print(f)

        if check_variable_dict(variableDict):
            print("File used")
            epsArray[j,0] = variableDict["epsA"]
            epsArray[j,1] = variableDict["epsC"]
            nummot = variableDict["nummot"]
            numsen = variableDict["numsen"]

            y = np.concatenate((y, variableDict["y"][:,:,0]), axis=0)
            x = np.concatenate((x, variableDict["x"][:,:,0]), axis=0)
            #c = np.concatenate((c, np.ones(1000) * ((variableDict["epsA"] - variableDict["epsC"])))) # difference
            #c = np.concatenate((c, np.ones(1000) * (variableDict["epsA"] * variableDict["epsC"]))) # product


            MspecDict[j] = [sig.spectrogram(variableDict["y"][:,motorChannel,0], fs=20.0, nperseg = nperseg, noverlap = noverlap) for motorChannel in range(nummot)]
            SspecDict[j]= [sig.spectrogram(variableDict["x"][:,sensorChannel,0], fs=20.0, nperseg = nperseg, noverlap = noverlap) for sensorChannel in range(numsen)]

            allSpecs = np.concatenate((allSpecs, concatenate_all_chanels(MspecDict[j], SspecDict[j])), axis = 1)
            thisSpecSum = sum_all_channels_and_frames(MspecDict[j], SspecDict[j])

            allSpecsSum = allSpecsSum + thisSpecSum
            dominantFrequencies[j] = np.argmax(thisSpecSum)
            #for motorChannel in range(nummot):
            #    allSpecs = np.concatenate((allSpecs, MspecDict[j][sensorChannel][2].T), axis = 1)
            #for sensorChannel in range(numsen):
            #    allSpecs = np.concatenate((allSpecs, Sspecdict[j][sensorChannel][2].T), axis = 1)

            j += 1
            if j >= 10 and args.test: break
        else:
            print("file not used")

        i += 1

    # cut to length
    epsArray = epsArray[:j,:]
    dominantFrequencies = dominantFrequencies[:j]

    # transform allSpecs so that the rows are different timeframes and the columns are different frequencys and modularities
    allSpecs = allSpecs.T


    #print("allspecssum shape", allSpecsSum.shape)

    # save overview file
    pickle.dump(overviewDict, open(pickleFolder + overviewFile, "wb"))
    print("overview pickle saved.")

    print(("allSpecs shape", allSpecs.shape))


    kmeansClusters = 6
    kmeans = KMeans(n_clusters=kmeansClusters, random_state=1)
    kmeans.fit(allSpecs)

    #print(kmeans.cluster_centers_.shape)

    pca1 = decomposition.PCA(n_components=2)

    # from sklearn.manifold import TSNE
    # pca1 = TSNE(n_components=2, random_state=0)
    # np.set_printoptions(suppress=True)
    pca1.fit(allSpecs)
    print(dir(pca1))

    # print clusters
    if False:
        plt.figure()
        cen_pca = pca1.transform(kmeans.cluster_centers_)
        allSpecs_pca = pca1.transform(allSpecs)

        c1i = 0
        c2i = 1
        plt.plot(allSpecs_pca[:,c1i], allSpecs_pca[:,c2i], "ko", alpha=0.4)
        plt.plot(cen_pca[:,c1i], cen_pca[:,c2i], "o", c=[0,0.25,0.5,0.75], ms=20)
        plt.title("CLUSTERZ")


    detailViews = np.random.randint(j, size=(4,4))
    for jn in range(j):
        prediction = kmeans.predict(concatenate_all_chanels(MspecDict[jn],SspecDict[jn]).T)
        behavior = prediction[np.argmax(np.bincount(prediction))]
        #print prediction, behavior

        c = np.concatenate((c, np.ones(1000) * behavior))
        ckl = ckl + [behavior]

    print(ckl)
    print("PCA")

    pca = decomposition.PCA(n_components=2)
    pca.fit(y)
    y = pca.transform(y)

    pca = decomposition.PCA(n_components=2)
    pca.fit(x)
    x = pca.transform(x)
    print("PCA finished")

    print(len(epsArray[:,0]))
    print("len clk %d" % len(ckl))

    # FIXME, why here
    from matplotlib import cm

    frequencies = MspecDict[0][0][0]
    cm = cm.get_cmap('Accent', max(c))

    # average dominant frequency
    if True:
        plt.figure()
        #normalize
        allSpecsSum /= np.sum(allSpecsSum)
        # from the first file get the first channel spectrogram and it's frequencies

        maxFreqIdx = np.argmax(allSpecsSum)

        plt.plot(frequencies, allSpecsSum)
        plt.plot([frequencies[maxFreqIdx]]*2, [0,allSpecsSum[maxFreqIdx]])
        plt.text(frequencies[maxFreqIdx] + 0.1, allSpecsSum[maxFreqIdx] / 3, "%.2f Hz" % frequencies[maxFreqIdx])

        plt.xlabel("Hz")
        plt.ylabel("percentage")
        plt.title("Dominant Frequencies")

    # heatmap of dominant frequencies
    if True:
        fig = plt.figure()
        plt.subplot(221)
        # normalize

        #dominantFrequencies /= np.max(dominantFrequencies)
        print("dominantFrequencies", dominantFrequencies)

        #ax1 = fig.add_subplot(gs[0,0])
        plt.title("Frequency Heatmap")
        plt.scatter(np.log(epsArray[:,0]), np.log(epsArray[:,1]), c=dominantFrequencies)
        plt.colorbar()

        plt.subplot(222)

        plt.title("Frequency Heatmap")
        plt.hexbin(np.log(epsArray[:,0]), np.log(epsArray[:,1]), C=dominantFrequencies, gridsize = 8, bins=None)
        plt.colorbar()

        plt.subplot(223)
        plt.hist(dominantFrequencies, bins=20)




    # epsC, epsA, class
    if False:
        gs = gridspec.GridSpec(2,2)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[0,0])
        ax1.scatter(epsArray[:len(ckl),0], epsArray[:len(ckl),1], c = ckl, s=80, alpha=1, edgecolors='none', vmin = 0, vmax = max(c), cmap = cm)
        #ax1.set_xscale("log")
        #ax1.set_yscale("log")
        plt.xlabel("epsA")
        plt.ylabel("epsC")
        #plt.show()
        #c -= min(c)
        #c /= max(c)
        #c *= 255.0

        ax2 = fig.add_subplot(gs[1,0])
        ax2.scatter(y[:,0],y[:,1], c=c, alpha = 0.07, edgecolors='none', vmin = 0, vmax = max(c),cmap = cm)

        ax3 = fig.add_subplot(gs[1,1])
        ax3.scatter(x[:,0],x[:,1], c=c, alpha = 0.07, edgecolors='none', vmin = 0, vmax = max(c),cmap = cm)

        fig = plt.figure()
        gs = gridspec.GridSpec(4,4)
        for x  in range(4):
            for y in range(4):
                ax = fig.add_subplot(gs[x,y])
                j = detailViews[x,y]
                plt.hist(kmeans.predict(concatenate_all_chanels(MspecDict[j],SspecDict[j]).T), range=(0,kmeansClusters-1))
                plt.title("EpsA: %.2f, EpsC: %.2f" % (epsArray[j,0], epsArray[j,1]))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pickle file crawler")
    parser.add_argument("-t", "--test", help="quicktest", action="store_true")
    parser.add_argument("-b", "--battery_in", type=bool, default=True)
    args = parser.parse_args()

    crawlFiles(args)
