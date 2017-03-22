import cPickle as pickle
import warnings
import time, argparse, sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lpzrobots ROS controller: test homeostatic/kinetic learning")
    parser.add_argument("file", type=str, help="filename (no default)", nargs='?')
    parser.add_argument("-w", "--windowsize", type=int, help="correlation window size", default=100)
    args = parser.parse_args()



filename = args.file
try:
    variableDict = pickle.load(open( filename, "rb" ) )
except Exception:
    raise Exception("File not found, use -f and the filepath")

if variableDict["dataversion"] == 1:
    warnings.warn("Pickle from V1, xsi might not be correct")
nummot = variableDict["nummot"]
motorCommands = variableDict["y"]
timesteps = len(motorCommands)
loopsteps = timesteps - args.windowsize

motorCommands += np.random.normal(size=motorCommands.shape) * 0.2

# testdata
# motorCommands = np.arange(20).reshape(10,2)
# print motorCommands

# print(len(motorCommands))


allCorrelations = np.zeros((loopsteps, nummot, nummot))
allAverageSquaredCorrelations = np.zeros((loopsteps, nummot * nummot))

for i in range(len(motorCommands) - args.windowsize):
    # use only the latest step of the buffer but all motor commands
    # shape = (timestep, motorChannel, buffer)

    window = motorCommands[i:i+args.windowsize,:,0]


    correlations = np.corrcoef(window.T)
    allCorrelations[i,:,:] = correlations[:,:]

    # save average of the squared upper triangle of the correlations
    allAverageSquaredCorrelations[i,:] = np.sum(np.triu(correlations,k=1).flatten() ** 2)

plt.figure()
plt.plot(allAverageSquaredCorrelations, alpha = 0.7)
#plt.plot(np.triu(correlations[:],k=1).flatten())
plt.show()
#print allAverageSquaredCorrelations
