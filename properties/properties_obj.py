import os
from torch.utils import data

import utils.utils_global as utils_global
from data_loaders import Dataset_obj
from properties.properties_global import PropertiesGlobal

properties_global = PropertiesGlobal()


class PropertiesObj:
    def __init__(self):
        # CUDA for PyTorch
        self.device = properties_global.device
        self.with_uncertainty = properties_global.with_uncertainty
        self.dataset_name = properties_global.dataset_name

        self.input_width = 640
        self.input_height = 480

        self.out_width = 80
        self.out_height = 60

        self.fx = 585
        self.fy = 585


        self.cluster = False
        self.refresh_scene_ground_truth = False
        self.save_outputs_from_validation = True

        self.datanum = 1000
        self.lr = 1e-4
        self.batch_size = 1
        if self.dataset_name == 'chess':
            self.sequences_train = ['seq01', 'seq02', 'seq04', 'seq06']
        elif self.dataset_name == 'pumpkin':
            self.sequences_train = ['seq02', 'seq03', 'seq06', 'seq08']
        else:
            raise ValueError('invalid dataset name')

        self.sequences_validation = []
        self.n_epochs = 75


        if not self.cluster:
            # working path and data path
            working_path = os.getcwd()
            data_path = os.path.join(working_path, 'data/' + self.dataset_name)
            save_in_folder = os.path.join('net_state_repo/' + self.dataset_name + '/l1_training')
            save_in_folder = os.path.join(os.getcwd(), save_in_folder)
        else:
            # modified working path and data path for cluster
            # todo: here set paths for you own environment
            data_path = os.path.join(os.environ['TMPDIR'], 'dataset')
            filename = '7scenes_chess'
            utils_global.uncompress_dataset(data_path, filename + '.zip')
            data_path = os.path.join(data_path, filename)
            save_in_folder = os.path.join('net_state_repo/' + self.dataset_name + '/l1_training')
            save_in_folder = os.path.join(os.getcwd(), save_in_folder)

        self.data_path = data_path
        self.save_in_folder = save_in_folder

        os.makedirs(self.save_in_folder, exist_ok=True)




    def get_dataloaders(self):

        params_3d_scene_generator = {'data_path': self.data_path, 'input_width': self.input_width, 'input_height': self.input_height,
                                     'out_width': self.out_width, 'out_height': self.out_height,
                                     'data_num': self.datanum, 'fx': self.fx, 'fy': self.fy, 'refresh': self.refresh_scene_ground_truth}

        # generate scene graound truths
        generator = utils_global.SceneGroundTruthGenerator(**params_3d_scene_generator)
        # can be called on a list of sequences
        generator(self.sequences_train)
        generator(self.sequences_validation)

        # get all the paths needed for training
        training_inputs = []
        training_labels = []
        for sequence in self.sequences_train:
            [rgb_path, poses_path, depth_path, scene_path] = utils_global.getPaths(self.data_path, sequence)
            for i in range(self.datanum):
                training_inputs.append(os.path.join(rgb_path, utils_global.rgbname(i)))
                training_labels.append(os.path.join(scene_path, utils_global.scenename(i)))

        validation_inputs = []
        validation_labels = []
        for sequence in self.sequences_validation:
            [rgb_path, poses_path, depth_path, scene_path] = utils_global.getPaths(self.data_path, sequence)
            for i in range(self.datanum):
                validation_inputs.append(os.path.join(rgb_path, utils_global.rgbname(i)))
                validation_labels.append(os.path.join(scene_path, utils_global.scenename(i)))

        print('path to dataset generated')

        # declare data loader
        data_loader_params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': 0}
        # training set
        training_set = Dataset_obj(training_inputs, training_labels, shift = False)
        training_generator = data.DataLoader(training_set, **data_loader_params)

        # validation set
        validation_generator = None
        if len(self.sequences_validation) > 0:
            validation_set = Dataset_obj(validation_inputs, validation_labels)
            validation_generator = data.DataLoader(validation_set, **data_loader_params)

        return training_generator, validation_generator

