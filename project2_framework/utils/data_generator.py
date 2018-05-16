import numpy as np
import torch
import math

def generate_data(row_dimension, col_dimension):
    data_input = np.random.uniform(0,1,(row_dimension,col_dimension))
    indices = np.arange(0,row_dimension)
    # apply a function on indices if input belongs to circle assign 1 otherwise assign 0
    data_output = np.asarray(list(map(lambda index: inCircle(data_input[index]),indices)))
    input_tensor = torch.from_numpy(data_input)
    target_tensor = torch.from_numpy(data_output)
    new_target = []
    for e in target_tensor.float():
        t = [0,0]
        t[int(e)] = 1
        new_target.append(t)
    target_tensor = torch.Tensor(new_target)
    return input_tensor.float(), target_tensor.float()

# check if (x,y) point is inside circle or not
def inCircle(values):
    radius = 1 / (math.sqrt(math.pi))
    x_co = math.pow(values[0] - 0.5, 2)
    y_co = math.pow(values[1] - 0.5, 2)
    return 1 if x_co + y_co < (math.pow(radius,2) / 2) else 0