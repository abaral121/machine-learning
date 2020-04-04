import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


df = pd.read_csv('C:\\Users\\abhis\\Documents\\Duke University\\IDS 705 Machine Learning\\A3_Q1_data.csv')

scatter = plt.scatter(x = 'x1', y = 'x2', c = 'y', data = df) 
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Spread of X1 vs. X2')
plt.legend(handles=scatter.legend_elements()[0], labels=['y = 0', 'y = 1'])
plt.show()

class logistic_function:

    def __init__(self,x,y,intercept = True):
        if intercept:
            intercept = np.ones(x.shape[0]).reshape(-1,1)
            x_constant = np.concatenate((intercept, x), axis = 1)
            self.x = x_constant
            self.y = y
        else:
            self.x = x
            self.y = y
            pass


    def probability(self,w):
        prob = (1 / (1 + np.exp(-self.x@w.T)))
        return prob


    def cost(self, w):
        pred = self.probability(w)
        cost = -(np.sum(self.y*np.log(pred)+(1-self.y)*np.log(1-pred))) / self.y.shape[0]
        return cost


    def gradient_descent(self, learning_rate, stop_criteria):

        np.random.seed(705)
        w_init = np.random.rand(self.x.shape[1])


        diff = 1


        full_costs = []


        while diff >= stop_criteria:
            full_costs.append(self.cost(w_init))
            w_init_norm = np.linalg.norm(w_init)
            gradient = (self.x.T@(self.probability(w_init)-self.y)) /self.y.shape[0]
            w_new = w_init - learning_rate*gradient
            w_new_norm = np.linalg.norm(w_new)
            diff = abs(w_new_norm - w_init_norm)
            w_init = w_new
        return w_init, full_costs
    pass

# assisted by varun

X = df[['x1','x2']].to_numpy()
y = df['y'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


train_output = logistic_function(x_train,y_train)
test_output = logistic_function(x_test,y_test)

rates = [0.1, 0.25, 0.5,1]
stop_criteria = 1e-06

plt.figure(figsize = (15,7))
for eta in rates:
    w_train, train_costs = train_output.gradient_descent(eta, stop_criteria)
    w_test, test_costs = test_output.gradient_descent(eta, stop_criteria)
    plt.plot(train_costs, label = f'Train: {eta}')
    plt.plot(test_costs, label = f'Test: {eta}', linestyle = 'dashed')
    pass

plt.xlim(0,25)
plt.ylabel('Costs')
plt.xlabel('Iteration number')
plt.title('Learning weights vs. costs for logistic regression')
plt.legend()
plt.show()
