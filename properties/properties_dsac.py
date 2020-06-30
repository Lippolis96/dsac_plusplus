import torch
import os
from torch.utils import data

import utils.utils_global as utils_global
from data_loaders import Dataset_dsac

from properties.properties_global import PropertiesGlobal

properties_global = PropertiesGlobal()
device = properties_global.device
device_cpu = properties_global.device_cpu

class PropertiesDsac:
        
    def __init__(self):
        self.with_uncertainty = properties_global.with_uncertainty
        self.dataset_name = properties_global.dataset_name
        self.cluster = False
        self.save_outputs = True
        self.save_all_models = True
        self.text_log = True

        self.model_name = 'repro_loss_model.pth'
        if self.dataset_name == 'chess':
            self.sequences_train = ['seq01', 'seq02', 'seq04', 'seq06']
        elif self.dataset_name == 'pumpkin':
            self.sequences_train = ['seq02', 'seq03', 'seq06', 'seq08']
        else:
            raise ValueError('invalid dataset name')
        self.sequences_validation = []
        self.batch_size = 1
        self.datanum = 1000
        self.starting_epoch = 0
        self.max_epochs = 7
        self.number_hypotheses = 120
        self.num_top = 50
        self.number_refinement_iterations = 3
        self.pnp_misses_error = 100
        self.inlier_threshold = 8  # pixels
        self.learning_rate = 1e-6
        self.clipping_value = 1e-3

        # NOTE: does not exist a function that performs block diagonal copy
        self.camera_torch_cat = torch.zeros(self.number_hypotheses * 3, self.number_hypotheses * 3, device=device, dtype=torch.float)
        for i in range(self.number_hypotheses):
            self.camera_torch_cat[3 * i:3 * i + 3, 3 * i:3 * i + 3] = properties_global.camera_matrix_torch

        if not self.cluster:
            # working path and data path
            working_path = os.getcwd()
            data_path = os.path.join(working_path, 'data/' + self.dataset_name)
            save_in_folder = os.path.join('net_state_repo/' + self.dataset_name + '/dsac_training')
            save_in_folder = os.path.join(os.getcwd(), save_in_folder)
        else:
            # modified working path and data path for cluster
            # todo: here set paths for you own environment
            data_path = os.path.join(os.environ['TMPDIR'], 'dataset')
            filename = '7scenes_chess'
            utils_global.uncompress_dataset(data_path, filename + '.zip')
            data_path = os.path.join(data_path, filename)
            save_in_folder = os.path.join('net_state_repo/' + self.dataset_name + '/dsac_training')
            save_in_folder = os.path.join(os.getcwd(), save_in_folder)

        path_to_model = os.path.join('net_state_repo/' + self.dataset_name + '/repro_training')

        self.model_path = os.path.join(path_to_model, self.model_name)
        self.data_path = data_path
        self.save_in_folder = save_in_folder
        
        os.makedirs(self.save_in_folder, exist_ok=True)


    def get_dataloaders(self):

        # get all the paths needed for training
        training_inputs = []
        training_poses = []
        for sequence in self.sequences_train:
            [rgb_path, poses_path, _, scene_path] = utils_global.getPaths(self.data_path, sequence)
            for i in range(self.datanum):
                training_inputs.append(os.path.join(rgb_path, utils_global.rgbname(i)))
                training_poses.append(os.path.join(poses_path, utils_global.posename(i)))

        validation_inputs = []
        validation_poses = []
        for sequence in self.sequences_validation:
            [rgb_path, poses_path, _, scene_path] = utils_global.getPaths(self.data_path, sequence)
            for i in range(self.datanum):
                validation_inputs.append(os.path.join(rgb_path, utils_global.rgbname(i)))
                validation_poses.append(os.path.join(poses_path, utils_global.posename(i)))

        print('path to dataset generated')

        # declare data loader
        data_loader_params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': 0}
        # training set
        training_set = Dataset_dsac(training_inputs, training_poses)
        training_generator = data.DataLoader(training_set, **data_loader_params)

        # validation set
        validation_generator = None
        if len(self.sequences_validation) > 0:
            validation_set = Dataset_dsac(validation_inputs, validation_poses)
            validation_generator = data.DataLoader(validation_set, **data_loader_params)

        return training_generator, validation_generator