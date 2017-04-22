# Linear-Regression-with-Gradient-Descent
## Synopsis
The purpose of this code is to provide insight into how the gradient descent algorithm can be used for linear regression by finding the minimum of the total squared error function: 

<div align = "center">
<img src="https://github.com/pickus91/Linear-Regression-with-Gradient-Descent/blob/master/figure_4.png" align="center" height="15" width="500">
</div>

This code estimates the partial derivative of the squared error function with respect to slope *m* and y-intercept *b* for each iteration using difference quotients. Final estimates for *m* and *b* are given when either (1) the specified tolerance is reached or (2) the algorithm has cycled through a specified number of iterations. 

<div align = "center">
<img src="https://github.com/pickus91/Linear-Regression-with-Gradient-Descent/blob/master/figure_1.png" align="left" height="200" width="275"> 
<img src = "https://github.com/pickus91/Linear-Regression-with-Gradient-Descent/blob/master/figure_2.png" align="center" height="200" width="275"> 
<img src = "https://github.com/pickus91/Linear-Regression-with-Gradient-Descent/blob/master/figure_3.png" align="right" height="200" width="275"> 
</div>

## Prerequistes
* [NumPy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/)

## Code Example
```
    m, b, m_steps, b_steps, SE, numIters = gradient_descent(data, start_m = 0, start_b = 0, 
                                                            learningRate = 0.001, numIterations = 1000, 
                                                            tolerance = 0.0001)

```

## Licence
This code is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details





