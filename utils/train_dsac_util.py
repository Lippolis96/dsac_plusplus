import os
import torch

import utils.utils_global as util

from properties.properties_dsac import PropertiesDsac
from properties.properties_global import PropertiesGlobal

properties_global = PropertiesGlobal()
properties_dsac = PropertiesDsac()


def Rodrigues(rvec, num_hp, device):

    theta = torch.norm(rvec, dim=0)
    n = rvec/theta
    n_t = n.transpose(0,1)

    I = torch.eye(3, device=device, dtype = torch.float)
    i, j, z = I[0, :3].unsqueeze(0), I[1, :3].unsqueeze(0), I[2, :3].unsqueeze(0)
    i, j, z = torch.cat(num_hp * [i]), torch.cat(num_hp * [j]), torch.cat(num_hp * [z])

    col0 = torch.cross(n_t, i).reshape(3 * num_hp, 1)
    col1 = torch.cross(n_t, j).reshape(3 * num_hp, 1)
    col2 = torch.cross(n_t, z).reshape(3 * num_hp, 1)

    C_cat = torch.cat((col0, col1, col2), axis=1)
    I_cat = torch.cat(num_hp * [I])
    theta_cat = torch.cat(3 * [theta.unsqueeze(1)], dim=1)
    theta_cat = theta_cat.reshape(-1, 1)

    a = (n[0]* n).unsqueeze(2)
    b = (n[1] * n).unsqueeze(2)
    c = (n[2] * n).unsqueeze(2)
    nnt = torch.cat((a, b, c), dim=2)
    nnt = nnt.reshape(3, -1)
    nnt = nnt.transpose(0, 1)

    return torch.cos(theta_cat) * I_cat + torch.sin(theta_cat) * C_cat + (1.0 - torch.cos(theta_cat)) * nnt


def project_prediction(prediction_vectorized, R_world_to_cam_est, transl_w_to_cam_estimated, camera_matrix_torch, num_hp):
    transl_w_to_cam_estimated = transl_w_to_cam_estimated.transpose(0,1).reshape(-1, 1)
    reprojected_prediction = torch.mm(R_world_to_cam_est, prediction_vectorized) + transl_w_to_cam_estimated #[3*num_hp x num_pixels]


    # NOTE: does not exist a function that performs block diagonal copy
    K = properties_dsac.camera_torch_cat
    # this is used for testing the model
    K = K[:3*num_hp, :3*num_hp]

    reprojected_prediction = torch.mm(K, reprojected_prediction)
    reprojected_prediction = torch.transpose(reprojected_prediction, 0, 1).reshape(4800, num_hp, 3)
    repro_z = reprojected_prediction[:, :, 2]
    repro_z = repro_z.transpose(0, 1)
    reprojected_prediction = reprojected_prediction.transpose(2,0)

    return reprojected_prediction[:2] / repro_z  # [dim on image plane, hyp, pixel]


def get_truth(pose_label):
    R_cam_to_world_true = pose_label[0, :3, :3].to(device=properties_global.device, dtype=torch.float)
    t_cam_to_world_true = pose_label[0, :3, 3].to(device=properties_global.device, dtype=torch.float)
    return R_cam_to_world_true, t_cam_to_world_true # t_cam_to_world_true = camera center position in world frame, vector points to camera