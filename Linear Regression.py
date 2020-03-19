import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class Linear_Regression:
    
    def __init__(self):
        pass
    
    def create_train(self, mean, std_d, num):
        
        # Creating train dataset
        self.X1 = np.random.normal(mean, std_d, num).reshape(num, 1)
        self.X2 = np.random.normal(mean, std_d, num).reshape(num, 1)
        
        self.train = np.concatenate((self.X1, self.X2), axis = 1)
        pass
        
    def create_test(self, mean, std_d, num):
        
        # Creating test dataset
        self.Y1 = np.random.normal(mean, std_d, num).reshape(num, 1)
        self.Y2 = np.random.normal(mean, std_d, num).reshape(num, 1)
        
        self.test = np.concatenate((self.Y1, self.Y2), axis = 1)
        pass
        
    def fit(self):
        # deriving covaraince and std
        covariance = sum((self.X1 - np.mean(self.X1))*(self.X2 - np.mean(self.X2))) / len(self.X1)
        std_X1 = np.std(self.X1)
        std_X2 = np.std(self.X2)
        
        # Correlation
        self.r = covariance/(std_X1 * std_X2)        
        
        # Slope
        self.slope = covariance / sum((self.X1 - np.mean(self.X1))**2)
        
        # Intercept
        self.intercept = -self.slope * np.mean(self.X1) + np.mean(self.X2)
        pass  
        
    def summary(self):
        r_squared_train = self.r ** 2
        RSS_train = sum((self.X2 - (self.intercept + self.slope * self.X1))**2)
        RSS_test = sum((self.Y2 - (self.intercept + self.slope * self.Y1))**2)
        r_squared_test = 1 - (RSS_test / sum((self.Y2 - np.mean(self.Y2))**2))
        
        print('The RSS for train is {0:.2f}'.format(float(RSS_train)))
        print('The RSS for test is {0:.2f}'.format(float(RSS_test)))
        pass      

    def plotting(self):     
        plot_train = pd.DataFrame(self.train, columns = ['X1', 'X2'])
        ax = sns.regplot(x='X1', y='X2', data=plot_train, line_kws={'label':"y={}x+{}".format(self.slope,self.intercept)})
        plt.xlabel('Variable 1')
        plt.ylabel('Variable 2')
        plt.title('Training set')
        ax.legend()
        plt.show()
        pass
    
    
        
        
        
model = Linear_Regression()

model.create_train(5,1,100)
model.fit()
model.create_test(5,1,100)
model.summary()

model.plotting()




























