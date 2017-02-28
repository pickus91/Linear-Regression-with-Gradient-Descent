# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:01:40 2017

@author: picku
"""
"""Linear regression via gradient descent"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
style.use('ggplot')

def total_squared_error(data, m, b):
    """Total squared error between the data points and line
    y = mx + b SE_line = (y_1-(m*x_1+b))^2 + (y_2-(m*x_2)+b)^2 + ...+ 
    (y_n - (m*x_n + b))^2"""     
    total_SE = 0
    for data_point in data:
        SE = (data_point[1] - (m * data_point[0] + b)) ** 2
        total_SE += SE
    
    return total_SE / len(data)
    
def partial_difference_quotient(f, points, v, i, h):
    """compute the ith partial difference quotient of f w.r.t ith position of v"""
    
    #add h to just the ith element of v
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)] 
           
    return (f(points, w[0], w[1]) - f(points, v[0], v[1])) / h

def estimate_gradient(points, current_m, current_b):
    
    m = current_m
    b = current_b
    
    partial_wrt_m = partial_difference_quotient(total_squared_error, points, [m, b], i = 0, h = 0.0001)
    partial_wrt_b = partial_difference_quotient(total_squared_error, points, [m, b], i = 1, h = 0.0001)

    return partial_wrt_m, partial_wrt_b
    
def step(points, current_m, current_b, learningRate):
    
    partial_wrt_m, partial_wrt_b = estimate_gradient(points, current_m, current_b)
    
    
    next_m = current_m - (learningRate * partial_wrt_m)
    next_b = current_b - (learningRate * partial_wrt_b)
    
    t_val = [next_m - current_m, next_b - current_b]
     
    return next_m, next_b, t_val

def gradient_descent(points, start_m, start_b, learningRate, numIterations, tolerance):
    
    #initial starting points
    next_m = start_m
    next_b = start_b
    
    #keeping track of m and b steps
    m_steps = []
    b_steps = []
    total_SE = []
    #t_val_steps = []
    
    for i in range(numIterations):        
        
        numIters = 0
        [next_m, next_b, t_val] = step(points, next_m, next_b, learningRate)
        if abs(t_val[0]) >= tolerance and abs(t_val[1]) >= tolerance:            
            m_steps.append(next_m)
            b_steps.append(next_b)   
            total_SE.append(total_squared_error(points, next_m, next_b)) 
            numIters = i + 1
            #t_val_steps.append(t_val[0])
        else:
            break            
        
    return next_m, next_b, m_steps, b_steps, total_SE, numIters
    
#plotting line with slope m and intercept b
def create_line(m, b, x):
    """create line to fit sample data points using slope m and
    y-intercept b computed via gradient descent"""
    y = m * x + b
    return y
    
def run():    
    #Sample Data
    x = [i for i in range(10)]
    y = [-1, -5, -3, -7, -10, -8, -4, -9, -7, -11]
    plt.scatter(x, y)
    
    data = np.column_stack((x,y))
    m, b, m_steps, b_steps, SE, numIters = gradient_descent(data, start_m = 0, start_b = 0, learningRate = 0.001, numIterations = 1000, tolerance = 0.0001)
           
    line = [create_line(m, b, i) for i in range(min(x)-1, max(x)+1)]
    
    plt.subplot(311)
    plt.plot(line, label = 'y = {:.2f}x + {:.2f}'.format(m, b))
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression via Gradient Descent')
    plt.show()
    plt.legend()
    
    #plot showing estimate of m and b through each iteration 
    plt.figure(1)
    plt.subplot(312)
    plt.plot(m_steps, b_steps)
    plt.title('Steps')
    plt.xlabel('m')
    plt.ylabel('b')   
    
    #Linear regression progression 
    plt.subplot(313)
    plt.scatter(x, y)
    for i, j in zip(m_steps, b_steps):
        line = [create_line(i, j, k) for k in range(int(round(min(x))-1), int(round(max(x)+1)))]
        plt.plot(line)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Progression')
    
    #Total squared error vs Number of Iterations
    plt.figure(2)
    plt.plot(SE)
    plt.ylabel('Total Squared Error')
    plt.xlabel('Number of Iterations')
        
    fig = plt.figure(3)
    ax = fig.gca(projection = '3d')    

    m_plot = np.arange(m - 10, m + 10, 0.5)
    b_plot = np.arange(b - 10, b + 10, 0.5)        
        
    m_plot, b_plot = np.meshgrid(m_plot, b_plot)
    SE = np.array(total_squared_error(data, m_plot, b_plot))    
    
    #Set axis limits
    ax.set_zlim(np.min(SE), np.max(SE))
    ax.set_xlim(np.min(m_plot), np.max(m_plot))
    ax.set_ylim(np.min(b_plot), np.max(b_plot))
    
    #Set axis labels
    ax.set_xlabel('m (slope)')
    ax.set_ylabel('b (y-intercept')    
    ax.set_zlabel('SE (Total Squared Error')    
    
    ax.plot_wireframe(m_plot, b_plot, SE, color = 'grey')
    
    ax.scatter(m, b, total_squared_error(data, m, b), s = 200, c = 'r', marker = '*') #, label = 'm = {}, b = {}'.format(m,b))
    ax.text(m + 0.25, b + 0.25, total_squared_error(data, m, b), '(m = {:.2f}, b = {:.2f})'.format(m,b), size = 12, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.show()
       
    
if __name__ == '__main__':
    run()