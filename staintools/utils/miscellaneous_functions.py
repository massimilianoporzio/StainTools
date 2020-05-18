import numpy as np
import torch

def get_sign(x):
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    return A / np.linalg.norm(A, axis=1)[:, None]
