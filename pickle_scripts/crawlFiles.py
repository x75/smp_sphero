# IMPORTANT THIS HAS A BUG IN IT ABOUT THE DIMENSIONS OF allSpecs

import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


from sklearn import decomposition


pickleFolder = '../pickles/'
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
    print "overviewDict found"
    if(len(overviewDict) == len(files)):
        print "overviewDict is complete"
        overviewComplete = True
    else:
        print "overviewDict is not complete rebuilding"
        overviewDict = {}
    #print overviewDict
else:
    overviewDict = {}

epsArray = np.zeros((len(files), 2))
i = j = 0

y = np.zeros((0,4))
x = np.zeros((0,6))
c = np.zeros((0))
ckl = []

MspecDict = {}
SspecDict = {}
allSpecs = np.zeros((0,35))

def concatenate_all_chanels(Mspecs, Sspecs):

    specs = np.zeros((0,35))
    for Mspec in Mspecs:
        specs = np.concatenate((specs, Mspec[2]))
    for Sspec in Sspecs:
        specs = np.concatenate((specs, Sspec[2]))

    return specs

print "This is gonna take some time"
for f in files:
    filename = os.path.basename(f)
    if(overviewComplete):
        if overviewDict[filename]["timesteps"] != 1000 or overviewDict[filename]["dataversion"] < 3 or overviewDict[filename]["lag"] != 2:
            print "file skiped"
            continue

    print("Unpickling %d of %d" % ( i, len(files)))
    variableDict = pickle.load(open( f, "rb" ) )

    # fix for files without filename
    if not "filename" in variableDict:
        variableDict["filename"] = f

    saveToOverview(variableDict)

    if(not "dataversion" in variableDict):
        print f

    if variableDict["timesteps"] == 1000 and variableDict["dataversion"] >= 3 and variableDict["lag"] == 2:
        epsArray[j,0] = variableDict["epsA"]
        epsArray[j,1] = variableDict["epsC"]
        nummot = variableDict["nummot"]
        numsen = variableDict["numsen"]

        y = np.concatenate((y, variableDict["y"][:,:,0]), axis=0)
        x = np.concatenate((x, variableDict["x"][:,:,0]), axis=0)
        #c = np.concatenate((c, np.ones(1000) * ((variableDict["epsA"] - variableDict["epsC"])))) # difference
        #c = np.concatenate((c, np.ones(1000) * (variableDict["epsA"] * variableDict["epsC"]))) # product

        import scipy.signal as sig
        MspecDict[j] = [sig.spectrogram(variableDict["y"][:,motorChannel,0], fs=20.0, nperseg = 32) for motorChannel in range(nummot)]
        SspecDict[j]= [sig.spectrogram(variableDict["x"][:,sensorChannel,0], fs=20.0, nperseg = 32) for sensorChannel in range(numsen)]

        print("One Channel.shape: ", MspecDict[j][0][2].shape)
        print("All Channels.shape: " ,concatenate_all_chanels(MspecDict[j], SspecDict[j]).shape)

        allSpecs = np.concatenate((allSpecs, concatenate_all_chanels(MspecDict[j], SspecDict[j])))
        #for motorChannel in range(nummot):
        #    allSpecs = np.concatenate((allSpecs, MspecDict[j][sensorChannel][2].T), axis = 1)
        #for sensorChannel in range(numsen):
        #    allSpecs = np.concatenate((allSpecs, Sspecdict[j][sensorChannel][2].T), axis = 1)

        j += 1
        if j >= 50: break
    i += 1



# save overview file
pickle.dump(overviewDict, open(pickleFolder + overviewFile, "wb"))
print "overview pickle saved."

print("allSpecs shape", allSpecs.shape)

from sklearn.cluster import KMeans
import matplotlib.cm
import random

kmeansClusters = 10
kmeans = KMeans(n_clusters=kmeansClusters, random_state=1)
kmeans.fit(allSpecs)
plt.figure()
print(kmeans.cluster_centers_[0])
print(kmeans.cluster_centers_.shape)

pca1 = decomposition.PCA(n_components=2)

# from sklearn.manifold import TSNE
# pca1 = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
pca1.fit(allSpecs)
print dir(pca1)


# plt.scatter(kmeans.cluster_centers_[0],kmeans.cluster_centers_[1])
cen_pca = pca1.transform(kmeans.cluster_centers_)
allSpecs_pca = pca1.transform(allSpecs)
print("shape check", cen_pca.shape, allSpecs.shape)
c1i = 0
c2i = 1
plt.plot(allSpecs_pca[:,c1i], allSpecs_pca[:,c2i], "ko", alpha=0.4)
plt.plot(cen_pca[:,c1i], cen_pca[:,c2i], "o", c=[0,0.25,0.5,0.75], ms=20)
plt.title("CLUSTERZ")

detailViews = np.random.randint(j, size=(4,4))
for jn in range(j):
    prediction = kmeans.predict(concatenate_all_chanels(MspecDict[jn],SspecDict[jn]))
    behavior = prediction[np.argmax(np.bincount(prediction))]
    #print prediction, behavior

    c = np.concatenate((c, np.ones(1000) * behavior))
    ckl = ckl + [behavior]

print ckl
print "PCA"


pca = decomposition.PCA(n_components=2)
pca.fit(y)
y = pca.transform(y)

pca = decomposition.PCA(n_components=2)
pca.fit(x)
x = pca.transform(x)
print "PCA finished"

print len(epsArray[:,0])
print "len clk %d" % len(ckl)

from matplotlib import gridspec, cm
cm = matplotlib.cm.get_cmap('rainbow')
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
        plt.hist(kmeans.predict(concatenate_all_chanels(MspecDict[j],SspecDict[j])), range=(0,kmeansClusters-1))
        plt.title("EpsA: %.2f, EpsC: %.2f" % (epsArray[j,0], epsArray[j,1]))
plt.show()
