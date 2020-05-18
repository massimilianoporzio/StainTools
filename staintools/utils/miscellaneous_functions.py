import numpy as np
import torch

def get_sign(x):
     
    if torch.cuda.is_available():
          return torch.sign(x)
    else:
          if x > 0:
               return +1
          elif x < 0:
               return -1
          elif x == 0:
               return 0
   
   


def normalize_matrix_rows(A):
    """
    Normalize the rows of an array.

    :param A: An array.
    :return: Array with rows normalized.
    """
    if torch.cuda.is_available():
          device = torch.device("cuda:0")
          A = torch.from_numpy(A).float().to(device)
          return A / A.norm(dim=1)[:, None]
    else:
          return A / np.linalg.norm(A, axis=1)[:, None]
