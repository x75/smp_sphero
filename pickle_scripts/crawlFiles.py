import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition

def all_files(directory):
    for path, dirs, files in os.walk(directory):
        for f in files:
            yield os.path.join(path, f)

files = [f for f in all_files("../pickles/")
    if f.endswith('.pickle')]

epsArray = np.zeros((len(files), 2))
i = j = 0

y = np.zeros((0,4))
x = np.zeros((0,6))
c = np.zeros((0))

print "This is gonna take some time"
for f in files:
    print("Unpickling %d of %d" % ( i, len(files)))
    variableDict = pickle.load(open( f, "rb" ) )

    if(not "dataversion" in variableDict):
        print f

    if variableDict["timesteps"] == 1000 and variableDict["dataversion"] >= 3 and variableDict["lag"] == 2:
        epsArray[j,0] = variableDict["epsA"]
        epsArray[j,1] = variableDict["epsC"]
        y = np.concatenate((y, variableDict["y"][:,:,0]), axis=0)
        x = np.concatenate((x, variableDict["x"][:,:,0]), axis=0)
        #c = np.concatenate((c, np.ones(1000) * ((variableDict["epsA"] - variableDict["epsC"]))) # difference
        c = np.concatenate((c, np.ones(1000) * (variableDict["epsA"] * variableDict["epsC"]))) # product
        j += 1
        #if j >= 10: break
    i += 1

print "PCA"

pca = decomposition.PCA(n_components=2)
pca.fit(y)
y = pca.transform(y)

pca = decomposition.PCA(n_components=2)
pca.fit(x)
x = pca.transform(x)
print "PCA finished"

#plt.figure()
#plt.scatter(epsArray[:,0], epsArray[:,1])
#plt.show()
c -= min(c)
c /= max(c)
c *= 255.0

plt.figure()
plt.scatter(y[:,0],y[:,1], c=c, alpha = 0.1, edgecolors='none')

plt.figure()
plt.scatter(x[:,0],x[:,1], c=c, alpha = 0.2, edgecolors='none')
plt.show()
