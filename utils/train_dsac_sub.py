import torch
import warnings

from random import shuffle

from BPnP import BPnP as BPnPsolver
import utils.train_dsac_util as ransac_util

from properties.properties_dsac import PropertiesDsac
from properties.properties_global import PropertiesGlobal

properties_dsac = PropertiesDsac()
properties_global = PropertiesGlobal()


def loss_pose(R_world_to_cam_est, transl_w_to_cam_estimated, R_cam_to_world_true, t_cam_to_world_true, num_hp, testing = False):
    # NOTE: there is not a function that performs block diagonal copy
    R_true_cat = torch.zeros(num_hp * 3, num_hp * 3, device=properties_global.device, dtype = torch.float)
    for i in range(num_hp):
        R_true_cat[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R_cam_to_world_true

    R_err = torch.mm(R_true_cat, R_world_to_cam_est)
    R_err = R_err.reshape(num_hp, 3, 3)
    arg = (R_err[:,0, 0] + R_err[:,1, 1] + R_err[:,2, 2] - 1) / 2
    thet = torch.abs(torch.acos(arg) / 3.14 * 180)

    # t_cam_to_world_true = posizione centro camera w.r.t. world frame
    t_cam_to_world_true = t_cam_to_world_true.to(device=properties_global.device, dtype=torch.float)
    # transl_w_to_cam_estimated = posizione centro camera w.r.t. camera frame

    t_cam_to_world_estimated = torch.zeros_like(transl_w_to_cam_estimated)

    for i in range(num_hp):
        t_cam_to_world_estimated[:, i] = -R_world_to_cam_est[i * 3: i * 3 + 3].t() @ transl_w_to_cam_estimated[:, i]

    error_t = torch.norm(t_cam_to_world_estimated.transpose(0, 1) - t_cam_to_world_true, dim=1)
    # from millimeters to centimeters
    error_t = error_t / 10


    # error angle degrees, translation in centimeters
    # error = thet + error_t
    error = torch.max(thet, error_t)

    if testing:
        return thet, error_t

    return error


v_coord = [i for i in range(properties_global.height_out)]
u_coord = [i for i in range(properties_global.width_out)]


def hyp_index(number_points):
    shuffle(v_coord)
    shuffle(u_coord)
    if(number_points == properties_global.width_out*properties_global.height_out):
        return properties_global.pixel_locations
    else:
        return torch.tensor([[u_coord[i], v_coord[i]] for i in range(number_points)])


def quantile(tensor, q):
    te = tensor.clone().detach()
    length = max(te.shape)
    te, _ = torch.topk(te, length)

    index = length - int(length * q)
    return te[index]


def getPoints_cpu(scene_coordinates, number_points):
    #get hyp
    #u,v coordinates
    P = hyp_index(number_points)
    #get scene points oss: u_coord = x_coord = column
    S = scene_coordinates[:, P[:, 1], P[:, 0]].transpose(1,0)
    P = (P * properties_global.subsampling).to(dtype=torch.float)
    P = P.reshape((1, number_points, 2))
    S = S.reshape((number_points, 3))
    return P,S


def get_point_u(prediction, uncertainty, npoints, threshold = .99, flag_sampling = 'proportional'):
    uncertainty = uncertainty.view(4800)
    prediction = prediction.view(3, 4800).transpose(1, 0)
    # get certainty
    certainty = 1/uncertainty
    if flag_sampling == 'uniform':
        # SAMPLE UNIFORMLY (ignore uncertainty, can be used to test non uncertain models)
        certainty = torch.ones_like(certainty)
    if flag_sampling == 'threshold':
        # ONLY SAMPLE FROM threshold TOP POINTS
        q = quantile(certainty, 1-threshold)
        certainty[certainty < q] = 0
    indexes = torch.multinomial(certainty, npoints)
    pts3d = prediction[indexes]
    pts2d = properties_global.pixel_locations[indexes].to(device=pts3d.device, dtype=float).unsqueeze(0) * properties_global.subsampling
    return pts2d, pts3d


def make_hp_cpu(prediction_cpu, uncertainty_cpu, number_hp, flag_sampling = 'proportional'):
    PnP_solver = BPnPsolver.apply
    #these should be already on cpu
    hp_success = 0
    hp_poses = torch.zeros((number_hp, 6), dtype = torch.float)
    counter_failed = 0
    not_called = True
    while(hp_success < number_hp):
        P, S = get_point_u(prediction_cpu, uncertainty_cpu, 4, flag_sampling=flag_sampling)
        try:
            pose_c = PnP_solver(P, S, properties_global.camera_matrix_torch_cpu)[0]
        except:
            counter_failed += 1
            if(counter_failed > properties_dsac.pnp_misses_error and not_called):
                #filename = str(uuid.uuid4()) + '.png'
                not_called = False
                #warnings.warn('PnP failed more than 100 times on this image\nimage will be saved as ' + filename, RuntimeWarning)
                warnings.warn('PnP failed more than 100 times on this image', RuntimeWarning)
                #ransac_util.save_as_image(prediction_cpu, filename)
            continue
        hp_poses[hp_success] = pose_c
        hp_success += 1
    return hp_poses, counter_failed


def compute_scores(poses, prediction_vectorized, num_hp):

    # calculate rotations and translations
    rot_vector_w_to_cam_estimated_cpu = poses[:, :3].transpose(0, 1).to(device=properties_global.device)
    transl_w_to_cam_estimated = poses[:, 3:].transpose(0, 1).to(device=properties_global.device)
    R_world_to_cam_est = ransac_util.Rodrigues(rot_vector_w_to_cam_estimated_cpu, num_hp,
                                               device=properties_global.device).to(device=properties_global.device)

    # calculate projection and reprojection error of prediction
    my_reprojected_prediction = ransac_util.project_prediction(prediction_vectorized, R_world_to_cam_est,
                                                               transl_w_to_cam_estimated,
                                                               properties_global.camera_matrix_torch, num_hp).transpose(0, 1)

    my_reprojection_error = (properties_global.true_pixel_coords - my_reprojected_prediction).norm(dim=1)


    scores = properties_dsac.inlier_threshold - my_reprojection_error

    return scores, R_world_to_cam_est, transl_w_to_cam_estimated


def get_losses_and_scores(prediction, uncertainty, num_hp, R_cam_to_world_true, t_cam_to_world_true, flag_sampling = 'proportional'):
    PnP_solver = BPnPsolver.apply

    prediction_vectorized = prediction.reshape(3, properties_global.width_out * properties_global.height_out)
    prediction_cpu = prediction.cpu()
    uncertainty_cpu = uncertainty.cpu()

    #initial poses
    #poses_cpu is the PnP output i.e. rotation is world to cam, translation is world to cam in camera frame
    poses_cpu, counter = make_hp_cpu(prediction_cpu, uncertainty_cpu, properties_dsac.number_hypotheses, flag_sampling = flag_sampling)

    scores_all, R_world_to_cam_est, transl_w_to_cam_estimated = compute_scores(poses_cpu, prediction_vectorized, num_hp)

    scores = torch.sigmoid(scores_all).sum(1)

    # run on GPU
    poses = poses_cpu.to(device=properties_global.device)
    pixel_locations = properties_global.pixel_locations

    camera_matrix = properties_global.camera_matrix_torch
    refinement_threshold = torch.topk(scores, properties_dsac.num_top)[0][properties_dsac.num_top - 1]


    for i in range(properties_dsac.number_refinement_iterations):
        #get the inliers (select only hypotheses we want and select pixels for which 10 - repro_error > 0
        # inliers_all = (thresh_minus_repro > 0)
        inliers_all = (scores_all > 0)
        for index, inliers in enumerate(inliers_all):
            #both P and S are on GPU
            P = pixel_locations[inliers]
            if(P.shape[0] < refinement_threshold or P.shape[0] < 5) : continue
            S = prediction[:, P[:, 1], P[:, 0]].transpose(1, 0)
            P = (P * properties_global.subsampling).to(dtype=torch.float)
            img = P.reshape((1, -1, 2))
            obj = S.reshape((-1, 3))

            try:
                poses[index,:] = PnP_solver(img, obj, camera_matrix, poses[index,:].unsqueeze(0))[0]
            except Exception as e:
                print('error BPnP during refinement')
                print(e)
                continue

        scores_all, R_world_to_cam_est, transl_w_to_cam_estimated = compute_scores(poses, prediction_vectorized,
                                                                                   properties_dsac.number_hypotheses)

    scores = (torch.sigmoid(scores_all)).sum(1)
    losses = loss_pose(R_world_to_cam_est, transl_w_to_cam_estimated, R_cam_to_world_true, t_cam_to_world_true, num_hp)

    return losses, scores, poses.to(device=properties_global.device_cpu)

