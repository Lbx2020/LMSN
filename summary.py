#--------------------------------------------#
# This part of the code is used to see the network structure
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.LMSN import get_LMSN

if __name__ == "__main__":

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = get_LMSN(20).to(device)
    summary(m, input_size=(3, 300, 300))

