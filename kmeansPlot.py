import os
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 3:
	print('python kmeansPlot.py label_dir data_dir')
	sys.exit(1)

labels = []
data = []

# Read all labels from all files output by Spark 
for filename in sorted(os.listdir(sys.argv[1])):
	with open(sys.argv[1] + "/" + filename, 'r') as fin:
		for line in fin:
			labels.append(int(line))


# Read original features of the data 
for filename in sorted(os.listdir(sys.argv[2])):
	with open(sys.argv[2] + "/" + filename, 'r') as fin:
		for line in fin:
			features = np.array(list(map(lambda e : float(e), line.split(','))))
			data.append(features)
			#labels.append(int(line))

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

plt.scatter(data[labels == 0, 0], data[labels == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(data[labels == 1, 0], data[labels == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(data[labels == 2, 0], data[labels == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

plt.legend()
plt.show()