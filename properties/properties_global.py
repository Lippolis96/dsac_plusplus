import torch
import numpy as np


class PropertiesGlobal:
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.device_cpu = 'cpu'
        self.with_uncertainty = True
        self.dataset_name = 'chess'

        self.fx = 585
        self.fy = 585

        self.input_width = 640
        self.input_height = 480

        self.width_out = 80
        self.height_out = 60

        self.subsampling = self.input_width // self.width_out

        self.u0 = self.input_width // 2
        self.v0 = self.input_height // 2

        self.u = [[i for i in range(self.width_out)] for j in range(self.height_out)]
        self.v = [[j for i in range(self.width_out)] for j in range(self.height_out)]
        self.pixel_locations = torch.tensor([[[self.u[i][j], self.v[i][j]] for j in range(self.width_out)] for i in range(self.height_out)], device=self.device).view(self.height_out * self.width_out, 2)
        self.true_pixel_coords = self.pixel_locations.t() * self.subsampling

        # internal params matrix
        self.camera_matrix_torch = torch.tensor([[self.fx, 0, self.u0], [0, self.fy, self.v0], [0, 0, 1]]).to(device = self.device, dtype=torch.float)
        self.camera_matrix_np = np.array([[self.fx, 0, self.u0], [0, self.fy, self.v0], [0, 0, 1]], dtype=float)
        self.camera_matrix_torch_cpu = torch.tensor([[self.fx, 0, self.u0], [0, self.fy, self.v0], [0, 0, 1]]).to(dtype=torch.float)

