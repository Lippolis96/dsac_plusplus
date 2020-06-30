import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    '''
    FCN architecture for neural-guided scene coordiante regression.
    The network has two output heads: One predicting a 3d scene coordinate, and a 1d neural guidance weight (log probability).
    The network makes dense predictions, but the output is subsampled by a factor of 8 compared to the input.
    '''

    OUTPUT_SUBSAMPLE = 8

    def __init__(self):
        '''
        Constructor.
        '''
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)

        self.res3_conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.res3_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(512, 512, 3, 1, 1)

        # output head 1, scene coordinates
        self.fc1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.fc2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.fc3 = nn.Conv2d(512, 4, 1, 1, 0)


    def forward(self, inputs):
        '''
        Forward pass.
        inputs -- 4D data tensor (BxCxHxW)
        '''

        x = inputs - 127
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        res = self.res2_skip(res) + x

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = res + x

        # output head 1, scene coordinates
        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        res = self.fc3(sc)

        scene_coords = res[:, :3]
        uncertainty = res[:, 3]
        uncertainty = torch.abs(uncertainty)
        uncertainty = torch.sqrt(uncertainty) + 1

        return scene_coords, uncertainty