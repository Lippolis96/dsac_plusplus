#!/usr/bin/env python
import os
import torch
import utils_global
import cv2 as cv
import numpy as np

from model_star import Network as Net
import train_dsac_util as dsac_util
import train_dsac_sub as sub

from properties_dsac import PropertiesDsac
from properties_test import PropertiesTest, losses
from utils_global import Logger, ProgressBar
from properties_global import PropertiesGlobal

properties_global = PropertiesGlobal()
properties_dsac = PropertiesDsac()
properties_test = PropertiesTest()
logger = Logger()

properties_test.shuffle_images = True
properties_test.number_samples = 2000
# number of ransac hypotheses
properties_test.n_hyp = 256
# number of refinement iterations
properties_test.n_ref = 100

# Loading pre-trained model
model = Net()
model.load_state_dict(torch.load(properties_test.model_path, map_location = properties_global.device))
model.to(properties_global.device)
model.eval()

generator_test = properties_test.get_dataloaders()




with torch.set_grad_enabled(False):
    
    running_accuracy = 0
    counter = 0
    pb = ProgressBar('testing model', len(generator_test), decimal_places=2, step=0.01)

    for index, (local_batch, pose_label) in enumerate(generator_test):
        if (index != 0): pb(index, message='running accuracy: ' + utils_global.digits(running_accuracy / (index) * 100))

        # get scene coordinate prediction
        local_batch = local_batch.to(device=properties_global.device, dtype=torch.float)
        prediction, uncertainty = model(local_batch)
        prediction, uncertainty = prediction.squeeze(0), uncertainty.squeeze(0)
        prediction_cpu = prediction.cpu()
        prediction_vectorized = prediction.view(3, properties_global.width_out * properties_global.height_out)
        R_cam_to_world_true, t_cam_to_world_true = dsac_util.get_truth(pose_label)

        #RANSAC LOOP
        n_successes = 0
        max_ransac_score = 0
        rvec0_max_score, T0_max_score = None, None

        while n_successes < properties_test.n_hyp:
            pts2d, pts3d = sub.get_point_u(prediction_cpu, uncertainty, 4, threshold=properties_test.top_points_percentile, flag_sampling='proportional')
            pts2d = np.array(pts2d.transpose(0, 1).detach().cpu())
            pts3d = np.array(pts3d.detach().cpu())
            # call Ransac
            success, rvec0, T0, _ = cv.solvePnPRansac(objectPoints=pts3d, imagePoints=pts2d,
                                                      cameraMatrix=properties_global.camera_matrix_np, distCoeffs=None,
                                                      flags=cv.SOLVEPNP_P3P, confidence=(1 - 1e-8),
                                                      reprojectionError=4, iterationsCount=10000)
            if(not success):continue
            n_successes += 1
            poses_cpu = torch.from_numpy(np.concatenate((rvec0, T0)).transpose((1,0))).to(dtype=torch.float)

            scores_all, R_world_to_cam_est, transl_w_to_cam_estimated = sub.compute_scores(poses_cpu, prediction_vectorized, 1)
            scores_all -= properties_dsac.inlier_threshold
            scores_all += properties_test.threshold
            score = torch.sigmoid(scores_all).sum(1)

            if(score > max_ransac_score):
                max_ransac_score = score
                rvec0_max_score, T0_max_score = rvec0, T0
                inliers = (scores_all > 0).nonzero()[:,1].cpu().numpy()

        # END RANSAC LOOP
        # iteratively improve rvec, T
        rvec, T = np.copy(rvec0_max_score), np.copy(T0_max_score)
        pts2d, pts3d = sub.getPoints_cpu(prediction_cpu, 4800)
        pts2d = np.array(pts2d.transpose(0, 1).detach().cpu())
        pts3d = np.array(pts3d.detach().cpu())

        uncertainty = uncertainty.view(4800)
        # select top properties_test.top_points_percentile certain points (can be used later)
        q = sub.quantile(uncertainty, properties_test.top_points_percentile)
        top_points = (uncertainty < q)
        top_points = top_points.unsqueeze(0)

        # REFINEMENT
        for i in range(properties_test.n_ref):
            # result of PnP, rotation world to cam and translation world to cam in camera frame
            pose_final = np.concatenate((rvec, T))
            poses_cpu = torch.from_numpy(pose_final.transpose((1, 0))).to(dtype=torch.float)
            scores_all, R_world_to_cam_est, transl_w_to_cam_estimated = sub.compute_scores(poses_cpu, prediction_vectorized, 1)
            # this way the test and training thresholds can be different
            scores_all -= properties_dsac.inlier_threshold
            scores_all += properties_test.threshold

            # use intersection
            # inliers = ((scores_all > 0) * top_points).nonzero()[:, 1].cpu().numpy()
            # use only good points
            # inliers = (top_points).nonzero()[:, 1].cpu().numpy()

            # use normal inliers (works best usually more robust)
            inliers = (scores_all > 0).nonzero()[:, 1].cpu().numpy()
            score = torch.sigmoid(scores_all).sum(1)
            # select inliers
            pts2d_i = pts2d[inliers]
            pts3d_i = pts3d[inliers]

            _, rvec, T = cv.solvePnP(objectPoints=pts3d_i, imagePoints=pts2d_i, cameraMatrix=properties_global.camera_matrix_np,
                                           distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True,
                                           rvec=rvec, tvec=T)
        # END REFINEMENT

        # GET LOSSES
        loss_rot, loss_trasl = losses(rvec, T, R_cam_to_world_true, t_cam_to_world_true)
        if (properties_test.is_inlier(loss_rot.item(), loss_trasl.item())): running_accuracy += 1

        loss_rot_ransac, loss_trasl_ransac = losses(rvec0_max_score, T0_max_score, R_cam_to_world_true, t_cam_to_world_true)

        # SAVE SOME SAMPLE IMAGES
        if properties_test.save_test_outputs and counter < max([len(generator_test) / 20, 10]):
            counter += 1
            # prints losses: [refined loss, non refined loss]
            args = {'model name': properties_test.model_name,
                    'rot loss w-w/o refinement(deg)': [utils_global.digits(loss_rot.item()), utils_global.digits(loss_rot_ransac.item())],
                    'trasl loss w-w/o refinement (cm)': [utils_global.digits(loss_trasl.item()), utils_global.digits(loss_trasl_ransac.item())],
                    'inlier count w-w/o refinement': str([utils_global.digits(torch.sigmoid(scores_all).sum(1).item()), utils_global.digits(max_ransac_score.item())])}
            # save images for debugging
            os.makedirs(properties_test.save_path, exist_ok=True)
            utils_global.save_from_testing_with_params(prediction, properties_test.save_path, counter * 10, args)
            utils_global.save_from_testing_with_params(local_batch[0], properties_test.save_path, counter * 10 + 1, args)
            uncertainty_r = torch.zeros_like(uncertainty)
            uncertainty_r[top_points.squeeze(0)] = 155
            uncertainty_r = uncertainty_r.view(60,80)
            uncertainty_g = ((uncertainty / uncertainty.max()) * 255).view(60,80)
            uncertainty_b = torch.zeros_like(uncertainty)
            uncertainty_b[(scores_all > 0).squeeze(0)] = 255
            uncertainty_b = uncertainty_b.view(60,80)
            uncertainty = torch.stack((uncertainty_b, uncertainty_g, uncertainty_r))
            utils_global.save_from_testing_with_params(uncertainty, properties_test.save_path, counter * 10 + 2, args)

        # LOG RESULTS
        d = {properties_test.key_loss : float(loss_rot + loss_trasl), properties_test.key_loss_ransac : float(loss_rot_ransac + loss_trasl_ransac),
             properties_test.key_loss_rot_ref : float(loss_rot), properties_test.key_loss_rot_ransac : float(loss_rot_ransac),
             properties_test.key_loss_trasl_ref: float(loss_trasl), properties_test.key_loss_trasl_ransac : float(loss_trasl_ransac),
             properties_test.key_softmax_score : len(inliers)}
        logger(d)



    print('accuracy: ', utils_global.digits(running_accuracy / len(generator_test) * 100))
    print('\n')

    properties_test.print_results(logger)
