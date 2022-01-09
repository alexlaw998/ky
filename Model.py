import torch.nn as nn


class Net(nn.Module):
    def __init__(self, dim_img, dim_lbl):
        super(Net, self).__init__()
        self.fc = nn.Linear(dim_img, dim_lbl)

    def forward(self, x):
        """
        :param x:  data instance
        :return: regression result
        """
        return self.fc(x)
