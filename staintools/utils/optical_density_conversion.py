import numpy as np
import torch
def min_(value1, value2):
    if is_number(value1) and isinstance(value2, torch.Tensor):
        return torch.clamp(value2, max=value1)
    elif is_number(value2) and isinstance(value1, torch.Tensor):
        return torch.clamp(value1, max=value2)
    elif isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        return torch.min(value1, value2)

def max_(value1, value2):
    if is_number(value1) and isinstance(value2, torch.Tensor):
        return torch.clamp(value2, min=value1)
    elif is_number(value2) and isinstance(value1, torch.Tensor):
        return torch.clamp(value1, min=value2)
    elif isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        return torch.max(value1, value2)

def convert_RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    if torch.cuda.isa_available():
        device = torch.device("cuda:0")
        I = torch.from_numpy(I).float().to(device)
        mask = (I == 0.0)
        I[mask] = 1.0
        return max_(-1*torch.log(I/255.0),1e-6)
    else:
        mask = (I == 0)
        I[mask] = 1
        return np.maximum(-1 * np.log(I / 255), 1e-6)

    
    
    
def convert_OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB.

    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, "Negative optical density."
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)






