import matplotlib.pyplot as plt
import numpy as np

def count():
	path = raw_input("Input the file: ")
	file = open(path, 'r')
	minlen = int(raw_input("Input the length lower bound: "))
	maxlen = int(raw_input("Input the length upper bound: "))
	A = 0
	T = 0
	C = 0
	G = 0
	if (maxlen < minlen):
		return
	dist = [0 for i in range(maxlen - minlen + 1 if maxlen >= minlen else 0)]
	while (True):
		line = file.readline()
		if (line == ''):
			break
		dist[len(line) - minlen - 1] += 1
		A += line.count('A')
		T += line.count('T')
		C += line.count('C')
		G += line.count('G')
	file.close()
	x = [i for i in range(minlen, maxlen + 1)]
	print "distribution: ", dist
	plt.plot(x, dist)
	plt.figure(2)
	X = [1,2,3,4]
	Y = np.array([A, T, C, G])
	Y = Y / (1.0 * (A + T + C + G))
	plt.bar(X,Y,width = 0.35,facecolor = 'lightskyblue',edgecolor = 'white')
	plt.show()
	print "ATCG %: ", Y


def main():
	count()

if __name__ == "__main__":
	main()
