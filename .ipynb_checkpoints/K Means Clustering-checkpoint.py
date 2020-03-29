import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cluster

class KMeans:
    '''
    Perform k-means clustering.
    '''
    
    def __init__(self, num_clusters, convergence_threshold=0.05, max_steps=100):
        '''
        num_clusters : int, number of clusters
        '''
        self.num_clusters = num_clusters
        self.convergence_threshold = convergence_threshold
        self.max_steps = max_steps
        pass
    
    # fit
    def fit(self, X):
        '''
        Run k-means clustering on dataset X.
        X : numpy 2D array of floats, rows are observations and columns are features
        '''
        self.X           = X
        num_observations = self.X.shape[0]
        num_features     = self.X.shape[1]
        
        # initialize centroids
        self.centroids = self.X[np.random.randint(low=0, high=num_observations-1, size=self.num_clusters), :]     
        
        # update centroids
        #are_centroids_unstable = True
        n_steps   = 0
        while(n_steps <= self.max_steps):
            self.points_to_centroids_map = self.assign_points_to_centroids()
            centroids_updated            = self.update_centroids()
            
            # compute change in centroid movement
            differences = np.mean(centroids_updated, axis=1) - np.mean(self.centroids, axis=1)
            if(np.all(differences < self.convergence_threshold)):
                break
            
            self.centroids = centroids_updated
            
            n_steps += 1
            pass
        
        pass
    
    
    def get_clusters(self):
        '''
        Return a list of centroids that each point belong to.
        '''
        return self.points_to_centroids_map
    
    # transform
    def transform(self, X):
        ''' 
        Assign cluster to observations in X .
        '''
        return self.assign_points_to_centroids(X, self.centroids)
            
        
    # sse
    def get_sse(self):
        '''
        Return sum of squared errors.
        '''
        sse = 0
        
        # compute sse for each cluster
        for c in range(self.num_clusters):
            c_points = np.where(self.points_to_centroids_map == c)
            points   = self.X[c_points]
            
            # sum all the distances between a centroid and the points that belong to the centroid
            for point in points:
                sse += self.get_distance(self.centroids[c], point)
                pass
            pass
        
        return sse
            
        
    # visualize clusters
    def visualize_clusters(self):
        '''
        Plot X and color them by their clusters.
        '''
        
        # list containing centroids that the point in X belong to
        mapping         = self.points_to_centroids_map.reshape((-1, 1))
        
        # plot points in the vicinity of each centroid
        for centroid in range(self.num_clusters):
            plt.scatter(self.X[np.where(mapping == centroid), 0], self.X[np.where(mapping == centroid), 1], label='Cluster '+str(centroid), color=np.random.rand(3,))
            pass
        
        # plot centroids
        #plt.scatter(self.centroids[0], self.centroids[1], color=np.random.rand(3,), label='Cluster centroids')
        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Clusters')



    # helper functions
    def initialize_centroid(self):
        '''
        Initialize a centroid of dimension [1, num_features]. Each element of the centroid is randomly drawn from the minimum and maximum values of a feature.
        '''
        num_features    = self.X.shape[1]

        # compute min and max value in each feature
        feature_min_max = []
        for i in range(num_features):
            feature_min_max.append((np.min(self.X[:, i]), np.max(self.X[:, i])))
            pass

        # generate centroid: a random vector of size num_features where each number is drawn uniformly within the min and max of each feature
        centroid = []
        for i in range(len(feature_min_max)):
            centroid.append(np.random.uniform(feature_min_max[i][0], feature_min_max[i][1]))
            pass

        return centroid

    def assign_points_to_centroids(self):
        '''
        Return a list of length equal to the number of points in the data. Each element of list is the centroid that is closest to that point.
        '''
        points_to_centroids_map = [0 for i in range(self.X.shape[0])]
        
        
        for i, point in enumerate(self.X):
            point_to_centroid_distances = []
            for centroid in self.centroids:
                
                # compute distance between a point and a centroid
                point_to_centroid_distances.append(self.get_distance(centroid, point))
                pass
            
            # append the centroid number that is closest to the point
            points_to_centroids_map[i] = np.argmin(point_to_centroid_distances)
            pass

        return np.array(points_to_centroids_map)
        

    def update_centroids(self):
        '''
        Assign each centroid the mean of all the points in their vicinity.
        '''
        centroids_updated = np.zeros((self.centroids.shape[0], self.centroids.shape[1]))
        for c in range(self.centroids.shape[0]):
            c_points             = np.where(self.points_to_centroids_map == c)
            
            if(self.X[c_points].shape[0] == 0):
                # respawn centroids if no points present in their vicinity
                centroids_updated[c] = self.initialize_centroid()
                print('unexpected error')
                pass
            else:
                # update centroids if they have some points in their vicinity
                centroids_updated[c] = np.apply_along_axis(np.mean, 0, self.X[c_points])
            pass

        return centroids_updated

    def get_distance(self, a, b):
        '''
        Euclidean distance between two n-dimensional points.
        '''
        a = np.array(a)
        b = np.array(b)
        return np.sqrt(np.sum(np.power((a - b), 2)))
    
    pass

num_observations = 5000
num_features     = 2
num_blobs        = 2
X, y = datasets.make_blobs(n_samples=num_observations, n_features=num_features, centers=num_blobs, random_state=0)

# fit k-means using k found through elbow method
k = 2
model = KMeans(num_clusters=k)
model.fit(X)
model.visualize_clusters()

# sum of squared error
model.get_sse()

k_range = range(2, 10)
sse = []
for k in k_range:
    model = KMeans(num_clusters=k)
    model.fit(X)
    sse.append(model.get_sse())
    pass
plt.plot(k_range, sse)
plt.xlabel("K",)
plt.ylabel("Sum of squared errors")
plt.title("Elbow Curve")