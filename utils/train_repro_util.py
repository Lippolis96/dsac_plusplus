import torch
from random import shuffle


def project_mask_gaussian(pt3d_w, pose, camMat, xv, yv, max_pixel_error, mask_to_zero, uncertainty=None):

    H_world2cam = torch.inverse(pose)
    R_world2cam, t_world2cam = H_world2cam[:3, :3], H_world2cam[:3, -1]
    pt3d_w = pt3d_w.view(3, 60 * 80)
    uncertainty = uncertainty.view(1, 60 * 80).squeeze(0)
    mask_to_zero = mask_to_zero.view(3, 60 * 80)
    mask_to_zero = mask_to_zero[0]

    pt3d_c = R_world2cam @ pt3d_w
    pt3d_c = pt3d_c.transpose(0, 1) + t_world2cam
    pt3d_c = pt3d_c.transpose(0, 1)
    px, py, pz = camMat @ pt3d_c

    mask_to_one = (pz > 1)  # keep pixels with depth > 1mm to avoid division by 0
    px, py = px / pz, py / pz
    px_filtered, py_filtered = px[mask_to_one], py[mask_to_one]
    xv_filtered, yv_filtered = xv[mask_to_one], yv[mask_to_one]
    uncertainty_filtered = uncertainty[mask_to_one]

    mask_to_zero_filtered = mask_to_zero[mask_to_one]
    err_x, err_y = xv_filtered - px_filtered, yv_filtered - py_filtered
    Repro_error = torch.sqrt(err_x ** 2 + err_y ** 2)

    Repro_error = torch.min(Repro_error, max_pixel_error)

    error_classic = Repro_error.clone()

    error_uncertainty = (Repro_error**2/(2 * uncertainty_filtered**2)) + 2*torch.log(uncertainty_filtered*(2*3.141592653589793)**.5)

    zeros = torch.zeros_like(Repro_error)

    error_classic[mask_to_zero_filtered] = zeros[mask_to_zero_filtered]
    error_uncertainty[mask_to_zero_filtered] = zeros[mask_to_zero_filtered]

    pixel_used = (mask_to_zero==False).sum().item()
    loss_uncertainty = error_uncertainty.sum()
    loss_uncertainty /= pixel_used
    loss_classic = error_classic.sum()
    loss_classic /= pixel_used

    return loss_uncertainty, loss_classic

def divide_train_test_set(TRAINING_VALIDATION_SPLIT, input_path, label_path, pose_path):
    # Shuffle two lists with same order
    # Using zip() + * operator + shuffle()

    temp = list(zip(input_path, label_path, pose_path))
    shuffle(temp)
    shuffle(temp)

    input, labels, pose = zip(*temp)
    input = list(input)
    labels = list(labels)
    pose = list(pose)

    num_training = int(len(input) * TRAINING_VALIDATION_SPLIT)
    num_validation = len(input) - num_training

    training_input = input[:num_training]
    training_labels = labels[:num_training]
    training_pose = pose[:num_training]

    validation_inputs = input[num_training:]
    validation_labels = labels[num_training:]
    validation_pose = pose[num_training:]

    return training_input, training_labels, validation_inputs, validation_labels, training_pose, validation_pose




#
#
# def project_mask_reg(pt3d_w, pose, camMat, xv, yv, max_pixel_error, mask_to_zero, uncertainty = None):
#
#     H_world2cam = torch.inverse(pose)
#     R_world2cam, t_world2cam = H_world2cam[:3, :3], H_world2cam[:3, -1]
#     pt3d_w = pt3d_w.view(3, 60 * 80)
#     uncertainty = uncertainty.view(1, 60 * 80).squeeze(0)
#     mask_to_zero = mask_to_zero.view(3, 60 * 80)
#     mask_to_zero = mask_to_zero[0]
#
#     pt3d_c = R_world2cam @ pt3d_w
#     pt3d_c = pt3d_c.transpose(0, 1) + t_world2cam
#     pt3d_c = pt3d_c.transpose(0, 1)
#     px, py, pz = camMat @ pt3d_c
#
#     mask_to_one = (pz > 1)  # keep pixels with depth > 1mm to avoid division by 0
#     px, py = px / pz, py / pz
#     px_filtered, py_filtered = px[mask_to_one], py[mask_to_one]
#     xv_filtered, yv_filtered = xv[mask_to_one], yv[mask_to_one]
#     uncertainty_filtered = uncertainty[mask_to_one]
#
#     mask_to_zero_filtered = mask_to_zero[mask_to_one]
#     err_x, err_y = xv_filtered - px_filtered, yv_filtered - py_filtered
#     error = torch.sqrt(err_x ** 2 + err_y ** 2)
#
#     error = torch.min(error, max_pixel_error)
#
#     error_classic = error.clone()
#     error_uncertainty = (error/uncertainty_filtered) + 3*torch.log(2*uncertainty_filtered)
#
#     zeros = torch.zeros_like(error)
#
#     error_classic[mask_to_zero_filtered] = zeros[mask_to_zero_filtered]
#     error_uncertainty[mask_to_zero_filtered] = zeros[mask_to_zero_filtered]
#
#     pixel_used = (mask_to_zero == False).sum().item()
#     loss_uncertainty = error_uncertainty.sum()
#     loss_uncertainty /= pixel_used
#     loss_classic = error_classic.sum()
#     loss_classic /= pixel_used
#
#     return loss_uncertainty, loss_classic
#
# def project_mask_laplacian(pt3d_w, pose, camMat, xv, yv, max_pixel_error, mask_to_zero, uncertainty = None):
#
#     H_world2cam = torch.inverse(pose)
#     R_world2cam, t_world2cam = H_world2cam[:3, :3], H_world2cam[:3, -1]
#     pt3d_w = pt3d_w.view(3, 60 * 80)
#     uncertainty = uncertainty.view(1, 60 * 80).squeeze(0)
#     mask_to_zero = mask_to_zero.view(3, 60 * 80)
#     mask_to_zero = mask_to_zero[0]
#
#     pt3d_c = R_world2cam @ pt3d_w
#     pt3d_c = pt3d_c.transpose(0, 1) + t_world2cam
#     pt3d_c = pt3d_c.transpose(0, 1)
#     px, py, pz = camMat @ pt3d_c
#
#     mask_to_one = (pz > 1)  # keep pixels with depth > 1mm to avoid division by 0
#     px, py = px / pz, py / pz
#     px_filtered, py_filtered = px[mask_to_one], py[mask_to_one]
#     xv_filtered, yv_filtered = xv[mask_to_one], yv[mask_to_one]
#     uncertainty_filtered = uncertainty[mask_to_one]
#
#     mask_to_zero_filtered = mask_to_zero[mask_to_one]
#     err_x, err_y = xv_filtered - px_filtered, yv_filtered - py_filtered
#     error = torch.sqrt(err_x ** 2 + err_y ** 2)
#     error_a = torch.abs(err_x) + torch.abs(err_y)
#     error_a = torch.min(error_a, max_pixel_error)
#
#     error = torch.min(error, max_pixel_error)
#
#     error_classic = error.clone()
#     error_uncertainty = (error_a/uncertainty_filtered) + 2*torch.log(2*uncertainty_filtered)
#
#     zeros = torch.zeros_like(error)
#
#     error_classic[mask_to_zero_filtered] = zeros[mask_to_zero_filtered]
#     error_uncertainty[mask_to_zero_filtered] = zeros[mask_to_zero_filtered]
#
#     pixel_used = (mask_to_zero == False).sum().item()
#     loss_uncertainty = error_uncertainty.sum()
#     loss_uncertainty /= pixel_used
#     loss_classic = error_classic.sum()
#     loss_classic /= pixel_used
#
#     return loss_uncertainty, loss_classic
