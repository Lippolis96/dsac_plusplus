import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils import data
import matplotlib.pyplot as plt

import utils.utils_global as utils_global
from data_loaders import Dataset_dsac
import utils.train_dsac_sub as sub

from properties.properties_global import PropertiesGlobal
properties_global = PropertiesGlobal()


class PropertiesTest:
    def __init__(self):
        self.model_name = 'dsac_repro_loss_model.pth'
        model_folder = 'dsac_training'
        path_to_model = os.path.join(os.getcwd(), 'net_state_repo/' + properties_global.dataset_name + '/' + model_folder)
        self.model_path = os.path.join(path_to_model, self.model_name)
        self.data_path = os.path.join(os.getcwd(), 'data/' + properties_global.dataset_name)
        self.save_path = os.path.join('net_state_repo/test_output', self.model_name[:-4])
        if properties_global.dataset_name == 'chess':
            self.sequences_test = ['seq03', 'seq05']
        elif properties_global.dataset_name == 'pumpkin':
            self.sequences_test = ['seq01', 'seq07']
        else:
            raise ValueError('invalid dataset name')

        self.shuffle_images = True
        self.datanum = 1000
        self.save_test_outputs = True
        self.batch_size = 1

        # number of ransac hypotheses
        self.n_hyp = 256
        # threshold for ransac scores
        self.threshold = 10
        # number of refinement iterations
        self.n_ref = 100

        self.top_points_percentile = .325

        self.key_loss = 'loss'
        self.key_loss_ransac = 'loss ransac'
        self.key_loss_rot_ref = 'loss rot ref'
        self.key_loss_rot_ransac = 'loss rot ransac'
        self.key_loss_trasl_ref = 'loss trasl ref'
        self.key_loss_trasl_ransac = 'loss trasl ransac'
        self.key_softmax_score = 'softmax score'


    def print_results(self, logger):
        # PLOT RESULTS
        d = utils_global.digits
        print('mean total loss: ', d(logger.call(self.key_loss, np.mean)))
        print('mean softmax score: ', d(logger.call(self.key_softmax_score, np.mean)))
        print('mean rotation loss (deg): ', d(logger.call(self.key_loss_rot_ref, np.mean)))
        print('mean translation loss (cm): ', d(logger.call(self.key_loss_trasl_ref, np.mean)))

        print('\n')

        print('median loss rot (with and without ref)', [d(logger.call(self.key_loss_rot_ref, np.median)), d(logger.call(self.key_loss_rot_ransac, np.median))])
        print('median loss trasl (with and without ref)', [d(logger.call(self.key_loss_trasl_ref, np.median)), d(logger.call(self.key_loss_trasl_ransac, np.median))])
        print('median loss tot (with and without ref)', [d(logger.call(self.key_loss, np.median)), d(logger.call(self.key_loss_ransac, np.median))])

        print('\n')

        a = logger.get(self.key_loss) - logger.get(self.key_loss_ransac)
        print('percentage poses improved by refinement', d(np.mean(a <= 0) * 100))

        _ = plt.hist(a, bins=50)  # arguments are passed to np.histogram
        plt.title('poses improve with refinement')
        plt.xlabel('loss_refined - loss_non_refined')
        plt.ylabel('absolute frequency')
        plt.show()

    def get_dataloaders(self):

        # get all the paths needed for training
        test_inputs = []
        test_poses = []
        for sequence in self.sequences_test:
            [rgb_path, poses_path, _, _] = utils_global.getPaths(self.data_path, sequence)
            for i in range(self.datanum):
                test_inputs.append(os.path.join(rgb_path, utils_global.rgbname(i)))
                test_poses.append(os.path.join(poses_path, utils_global.posename(i)))
        print('path to dataset generated')

        # declare data loader
        data_loader_params = {'batch_size': self.batch_size, 'shuffle': self.shuffle_images, 'num_workers': 0}
        # training set
        test_set = Dataset_dsac(test_inputs, test_poses)
        test_generator = data.DataLoader(test_set, **data_loader_params)

        return test_generator

    @staticmethod
    def is_inlier(rot, trasl):
        return (rot < 5 and trasl < 5)


def losses(rotation_estimate, traslation_estimate, R_cam_to_world_true, t_cam_to_world_true):
    T = traslation_estimate
    rvec = rotation_estimate
    # this is the loss of the complete estimate (with refinement)
    transl_w_to_cam_estimated = torch.tensor(T, dtype=torch.float, device = properties_global.device)

    R_world_to_cam_est = R.from_rotvec(rvec[:, 0]).as_matrix()
    R_world_to_cam_est = torch.tensor(R_world_to_cam_est, dtype=torch.float, device = properties_global.device)

    loss_rot, loss_trasl = sub.loss_pose(R_world_to_cam_est, transl_w_to_cam_estimated, R_cam_to_world_true, t_cam_to_world_true, 1, testing = True)

    return loss_rot, loss_trasl