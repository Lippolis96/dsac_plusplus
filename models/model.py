import torch.nn as nn
import torch.nn.functional as F
import torch

class NetVanilla(nn.Module):

    def __init__(self):
        print('TORCH: Creating network.')
        super(NetVanilla, self).__init__()

        ''' initialization: nn.Conv2d(in_channels, out_channels, kernel_size,
                                        stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')'''
        # block 1:  -- 640 x 480
        self.conv1 = nn.Conv2d(3, 64, (3,3), (1, 1), (1,1))          # -- 3
        self.conv2 = nn.Conv2d(64, 128, (3,3), (2, 2), (1,1))        # -- 5

        # block 2 -- 320 x 240
        self.conv3 = nn.Conv2d(128, 128, (3,3), (2, 2), (1,1))       # -- 9

        # block 3 -- 160 x 120
        self.conv4 = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))     # -- 17
        self.conv5 = nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1))     # -- 19

        # block 4 -- 80 x 60
        self.conv6 = nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1))     # -- 37
        self.conv7 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))     # -- 39
        self.conv8 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))     # -- 41

        # block 5
        self.conv9 = nn.Conv2d(512, 4096, (1, 1), (1, 1), (0, 0))
        self.conv10 = nn.Conv2d(4096, 4096, (1, 1), (1, 1), (0, 0))
        self.conv11 = nn.Conv2d(4096, 4, (1, 1), (1, 1), (0, 0))

    def forward(self, x):
        ''' call to conv2d: nn.conv2d([batch_size, channels, height, width]) '''

        # normalization
        x = x - 127

        # forward block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        res = self.conv11(x)

        scene_coords = res[:, :3]
        uncertainty = res[:, 3]
        uncertainty = torch.abs(uncertainty)
        uncertainty = torch.sqrt(uncertainty) + 1

        return scene_coords, uncertainty