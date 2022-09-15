

import torch.nn as nn 
import numpy as np 

x= np.array([1,2,3,4])
y= np.array([2,4,6,8])


class LinearRegression: 
    def __init__(self,x,y,w=0): 
        self.x = x 
        self.y = y
        self.w = w  

    def forward(self): 
        return self.x*self.w 

    def loss(self,y_predicted): 
        return np.sum(self.y-y_predicted)**2

    def gradient( self,y_predicted): 
        return 2*np.dot(self.w, self.y, y_predicted).mean() 

    def train ( self,n_iterations, learning_rate): 
        for epoch in range(n_iterations): 
            y_predicted = self.forward(self)
            loss_linear = self.loss(self, y_predicted)
            dw = self.gradient(self, y_predicted)
            w-= dw*learning_rate

    def predictions(self,x_input): 
        return self.w*x_input


linear= LinearRegression(x,y)
linear.train(10,0.1)
print(linear.predictions(5))