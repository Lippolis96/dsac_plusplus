import torch
import os
import utils.utils_global as utils_global
from torch.utils import data

from data_loaders import Dataset_repro
from properties.properties_global import PropertiesGlobal

properties_global = PropertiesGlobal()


class PropertiesRepro:
    def __init__(self):
        # CUDA for PyTorch
        self.with_uncertainty = properties_global.with_uncertainty
        self.device = properties_global.device
        self.dataset_name = properties_global.dataset_name

        self.pixel_error_threshold = torch.tensor(100.0, device=self.device, dtype=torch.float)
        # training parameters
        self.save_outputs = True
        self.n_epochs = 75
        self.batch_size = 1
        self.lr = 0.0001
        self.mask_black_pixels = True

        # number of images for each sequence
        self.datanum = 1000

        self.cluster = False
        if self.dataset_name == 'chess':
            self.sequences_train = ['seq01', 'seq02', 'seq04', 'seq06']
        elif self.dataset_name == 'pumpkin':
            self.sequences_train = ['seq02', 'seq03', 'seq06', 'seq08']
        else:
            raise ValueError('invalid dataset name')
        self.sequences_validation = []
        self.model_name = 'l1_loss_model.pth'

        if not self.cluster:
            # working path and data path
            working_path = os.getcwd()
            data_path = os.path.join(working_path, 'data/' + self.dataset_name)
            save_in_folder = os.path.join('net_state_repo/' + self.dataset_name + '/repro_training')
            save_in_folder = os.path.join(os.getcwd(), save_in_folder)
        else:
            # modified working path and data path for cluster
            # todo: here set paths for you own environment
            data_path = os.path.join(os.environ['TMPDIR'], 'dataset')
            filename = '7scenes_chess'
            utils_global.uncompress_dataset(data_path, filename + '.zip')
            data_path = os.path.join(data_path, filename)
            save_in_folder = os.path.join('net_state_repo/' + self.dataset_name + '/repro_training')
            save_in_folder = os.path.join(os.getcwd(), save_in_folder)

        self.data_path = data_path
        self.save_in_folder = save_in_folder
        path_to_model = os.path.join('net_state_repo/' + self.dataset_name + '/l1_training')
        self.path_model_to_load = os.path.join(path_to_model, self.model_name)

        os.makedirs(self.save_in_folder, exist_ok=True)

    def get_dataloaders(self):

        # get all the paths needed for training
        training_inputs = []
        training_labels = []
        training_poses = []
        for sequence in self.sequences_train:
            [rgb_path, poses_path, _, scene_path] = utils_global.getPaths(self.data_path, sequence)
            for i in range(self.datanum):
                training_inputs.append(os.path.join(rgb_path, utils_global.rgbname(i)))
                training_labels.append(os.path.join(scene_path, utils_global.scenename(i)))
                training_poses.append(os.path.join(poses_path, utils_global.posename(i)))

        validation_inputs = []
        validation_labels = []
        validation_poses = []
        for sequence in self.sequences_validation:
            [rgb_path, poses_path, _, scene_path] = utils_global.getPaths(self.data_path, sequence)
            for i in range(self.datanum):
                validation_inputs.append(os.path.join(rgb_path, utils_global.rgbname(i)))
                validation_labels.append(os.path.join(scene_path, utils_global.scenename(i)))
                validation_poses.append(os.path.join(poses_path, utils_global.posename(i)))

        print('path to dataset generated')

        # declare data loader
        data_loader_params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': 0}
        # training set
        training_set = Dataset_repro(training_inputs, training_labels, training_poses)
        training_generator = data.DataLoader(training_set, **data_loader_params)

        # validation set
        validation_generator = None
        if len(self.sequences_validation) > 0:
            validation_set = Dataset_repro(validation_inputs, validation_labels, validation_poses)
            validation_generator = data.DataLoader(validation_set, **data_loader_params)

        return training_generator, validation_generator