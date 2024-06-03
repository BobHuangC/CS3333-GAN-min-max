import numpy as np
import matplotlib.pyplot as plt
import imageio
from copy import deepcopy

# For the convenience of visualization, 
# we only consider optimization function of 2 D


def GDA_update_step(alpha, nabla_x, nabla_y):
    """
    构造一个function, 这个function输入 x, y, 生成 x_{t+1}, y_{t+1}
    
    x_{t+1} = x_t - \alpha \nabla_x f(x_t, y_t)
    y_{t+1} = y_t + \alpha \nabla_y f(x_t, y_t)
    
    x_t, y_t are float type
    alpha is the learning rate
    nabla_x, nabla_y are the function that takes in the x_t and y_t that returns float
    
    return the next step of x and y
    """
    def update(x_t, y_t):
        x_t_plus_1 = x_t - alpha * nabla_x(x_t, y_t)
        y_t_plus_1 = y_t + alpha * nabla_y(x_t, y_t)
        return x_t_plus_1, y_t_plus_1
    return update