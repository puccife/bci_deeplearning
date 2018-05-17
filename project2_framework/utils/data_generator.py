import torch
import math


def generate_data(row_dimension, col_dimension):
    """
    generates the dataset as suggested in the assignment
    """
    input_tensor = torch.Tensor(row_dimension, col_dimension).uniform_(0, 1)
    target_tensor = torch.arange(0, row_dimension)
    target_tensor.apply_(lambda index: 1 if inCircle(input_tensor[int(index)]) else 0)
    new_target = []
    for e in target_tensor.float():
        t = [0, 0]
        t[int(e)] = 1
        new_target.append(t)
    target_tensor = torch.Tensor(new_target)
    return input_tensor.float(), target_tensor.float()

def inCircle(values):
    """
    checks if (x,y) point is inside circle or not
    """
    radius = 1 / (math.sqrt(math.pi))
    x_co = math.pow(values[0] - 0.5, 2)
    y_co = math.pow(values[1] - 0.5, 2)
    return 1 if x_co + y_co < (math.pow(radius, 2) / 2) else 0
