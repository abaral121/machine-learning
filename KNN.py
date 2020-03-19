import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

A2_X_train_low = pd.read_csv("C:\\Users\\abhis\\Documents\\Duke University\\IDS 705 Machine Learning\\A2_X_train_low.csv", header = None).to_numpy()
A2_X_test_low = pd.read_csv("C:\\Users\\abhis\\Documents\\Duke University\\IDS 705 Machine Learning\\A2_X_test_low.csv", header = None).to_numpy()
A2_y_train_low = pd.read_csv("C:\\Users\\abhis\\Documents\\Duke University\\IDS 705 Machine Learning\\A2_y_train_low.csv", header = None).to_numpy()
A2_y_test_low = pd.read_csv("C:\\Users\\abhis\\Documents\\Duke University\\IDS 705 Machine Learning\\A2_y_test_low.csv", header = None).to_numpy()

A2_X_train_high = pd.read_csv("C:\\Users\\abhis\\Documents\\Duke University\\IDS 705 Machine Learning\\A2_X_train_high.csv", header = None).to_numpy()
A2_X_test_high = pd.read_csv("C:\\Users\\abhis\\Documents\\Duke University\\IDS 705 Machine Learning\\A2_X_test_high.csv", header = None).to_numpy()
A2_y_train_high = pd.read_csv("C:\\Users\\abhis\\Documents\\Duke University\\IDS 705 Machine Learning\\A2_y_train_high.csv", header = None).to_numpy()
A2_y_test_high = pd.read_csv("C:\\Users\\abhis\\Documents\\Duke University\\IDS 705 Machine Learning\\A2_y_test_high.csv", header = None).to_numpy()


class Knn:
# k-Nearest Neighbor class object for classification training and testing
    def __init__(self):
        # Initialize x and y
        pass
        
    def fit(self, x, y):
        # Save the training data to properties of this class
        self.x = x
        self.y = y
        pass
        
    def predict(self, x_test, k):
        
        # To store final classifcations
        self.y_pred = []
        self.x_test = x_test
        self.k = k
        pass
        
        for row in self.x_test:
            # measuring distance from row to all values of train
            dist = np.sqrt(np.sum((row - self.x)**2, axis = 1))
            
            # sorting and returning index numbers up to k
            dist_ind = np.argsort(dist)[0:k]
            y_ind = np.int8(self.y[dist_ind])
           
            # getting the common class
            class_mode = int(stats.mode(y_ind)[0])
            
            # appending outcomes to a list
            self.y_pred.append(class_mode)
        
        self.y_pred = np.asarray(self.y_pred)
        pass
            
    def accuracy(self, y):
        # simple classification accuracy 
        accuracy = sum(np.int8(y).reshape(-1,1) == self.y_pred.reshape(-1,1)) / len(y)
        return accuracy
        pass
        
    def plotting(self):
        self.x_test = pd.DataFrame(self.x_test, columns = ['X1', 'X2'])
        self.y_pred = pd.DataFrame(self.y_pred, columns = ['Class'])
        full = pd.concat([self.x_test, self.y_pred], axis=1) 
        sns.scatterplot(x='X1', y='X2', hue='Class', style = 'Class' ,data=full)
        plt.xlabel('Variable 1')
        plt.ylabel('Variable 2')
        plt.title('Test Set Predictions with k = {}'.format(self.k))
        pass
    
    
model = Knn()

model.fit(A2_X_train_low, A2_y_train_low)
model.predict(A2_X_test_low, 3)
model.accuracy(A2_y_test_low)
model.plotting()




