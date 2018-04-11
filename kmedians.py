import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self,data,numpoints):
	    
        self.stream = self.data_stream(data)
        self.numpoints = numpoints
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream)
        self.scat = self.ax.scatter(x, y, c=c, s=s, animated=True)
        self.ax.axis([-10, 10, -10, 10])

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self,data):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        
        xy = data[:2, :]
        s, c = data[2:, :]
        xy -= 0.5
        xy *= 10
        while True:
            xy += 0.03 * (np.random.random((2, self.numpoints)) - 0.5)
            s += 0.05 * (np.random.random(self.numpoints) - 0.5)
            c += 0.02 * (np.random.random(self.numpoints) - 0.5)
            yield data

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:2, :])
        # Set sizes...
        self.scat._sizes = 300 * abs(data[2])**1.5 + 100
        # Set colors..
        self.scat.set_array(data[3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def show(self):
        plt.show()
		
#Choosing random centers
def cluster_centers(K,data):
	centers_chosen = np.zeros(shape=(K,2),dtype='int32')
	for i in range(K):
		centers_chosen[i] = data[np.random.choice(len(data))]
	return centers_chosen	
		
def costTable(N,K,data,initial_centers):
	cost = []
	temp = {}
	for i in range(K):	
		for j in range(N):
			cost.append(sum(abs(data[j] - initial_centers[i])))
		temp[str(i)] = cost
		cost = []	
	return temp		
	
def caculate_total_cost(cost,N,K):
	costTotal = 0
	c = dict()
	temp = []
	
	for i in range(N):
		for j in range(K):
			#if cost[str(j)][i]!=0:
			temp.append(cost[str(j)][i])
		ind = temp.index(min(temp))
		if ind in c:
			c[ind].append(data[i])
		else:
			c[ind]=[data[i]]
		
		costTotal = costTotal + min(temp)
		temp = []
		
	return c,costTotal

	
def plot_points(c,K,data,centers):
	
	#Plot of data
	fig = plt.figure(1)
	ax = fig.add_subplot(1, 1, 1)	
	x = data[:,0]
	y = data[:,1]
	ax.scatter(x, y)
	s = 121
	
	for i in range(K):
		c_arr = np.array(c[i])
		ax = fig.add_subplot(1, 1, 1)
		c_arr_x = c_arr[:,0]
		c_arr_y = c_arr[:,1]
		ax.scatter(c_arr_x, c_arr_y)	
	
	c_x = centers[:,0]
	c_y = centers[:,1]
	ax.scatter(c_x,c_y,color='#000000',s = 100,marker='d',alpha=0.5)
		
	plt.show()
	

#Number of points
N = 70

#Number of clusters
K = 3

#Generating random data of N points
data = np.random.choice(60, size=(N, 2))
costTrack = {}

#No of iterations
iter = 100

for i in range(iter):
    #print('\n Data : \n',data,'\n')
	centers = cluster_centers(K,data)
	#print('\n Centers : \n',centers,'\n')
	cost = costTable(N,K,data,centers)
	#print(cost)
	c,costTotal = caculate_total_cost(cost,N,K)
	#plot_points(c,K,data)
	costTrack[costTotal] = centers
	
min_cost_index = min(list(costTrack.keys()))
print('Cluster Centers having minimum cost of '+str(min_cost_index)+' units is \n',costTrack[min_cost_index])	
	
cost = costTable(N,K,data,costTrack[min_cost_index])		
c,costTotal = caculate_total_cost(cost,N,K)
plot_points(c,K,data,costTrack[min_cost_index])
