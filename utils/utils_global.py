import os
import numpy as np
import torch
import cv2
from zipfile import ZipFile
import time
import datetime
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def get_pixel_ground_truth():
    yv, xv = torch.meshgrid([torch.arange(0, 60), torch.arange(0, 80)])
    yv, xv = yv.reshape(4800), xv.reshape(4800)
    yv, xv = yv * 8, xv * 8
    return xv, yv


def digits(number, digits=2):
    ff = '{:.' + str(digits) + 'f}'
    return ff.format(number)


def l1_loss(input, target):
    input = input.view(3, 60 * 80).transpose(0, 1)
    target = target.view(3, 60 * 80).transpose(0, 1)
    # -- loss is the Euclidean distance between predicted and ground truth coordinate, mean calculated over batch
    loss = torch.norm(input - target, 1, 1)
    loss = torch.mean(loss[torch.isfinite(loss)])
    return loss


def quantile(tensor, q):
    te = tensor.clone().detach()
    length = max(te.shape)
    te, _ = torch.topk(te, length)

    index = length - int(length * q)
    return te[index]


def l1_loss_u(input, target, uncertainty):
    input_c = input[0]
    input_c = input_c.view(3, 60 * 80).transpose(0, 1)
    target = target.view(3, 60 * 80).transpose(0, 1)

    input_u = uncertainty[0]
    input_u = input_u.view(1, 60 * 80).transpose(0, 1).squeeze(1)

    # -- loss is the Euclidean distance between predicted and ground truth coordinate, mean calculated over batch
    loss_c = torch.norm(input_c - target, 1, 1)
    M = torch.isfinite(loss_c)
    loss_c = loss_c[M]
    loss_c_w = loss_c / input_u[M]

    loss_u = torch.log(2 * input_u[M])
    loss_u = loss_c_w + 3*loss_u

    loss_u = torch.mean(loss_u)
    loss_L1 = torch.mean(loss_c).detach()

    return loss_u, loss_L1


def uncompress_dataset(path_extraction, filename, path_to_file='/cluster/work/riner/users/glippoli/datasets'):
    path = os.path.join(path_to_file, filename)
    if not os.path.exists(path):
        print('Dataset not found')
    with ZipFile(path, 'r') as zipObj:
        zipObj.extractall(path_extraction)


def getPaths(data_path, sequence):
    sequence_path = os.path.join(data_path, sequence)
    # read
    rgb_path = os.path.join(sequence_path, 'rgb')
    poses_path = os.path.join(sequence_path, 'poses')
    depth_path = os.path.join(sequence_path, 'depth')
    # write
    scene_path = os.path.join(sequence_path, 'scene')
    return [rgb_path, poses_path, depth_path, scene_path]


def withleading(mint):
    return str(f"{mint:06d}")


def depthname(sequence, index):
    return sequence + '_frame-' + str(withleading(index)) + '.pose.depth.tiff'


def posename(index):
    return 'frame-' + str(withleading(index)) + '.pose.txt'


def rgbname(index):
    return 'frame-' + str(withleading(index)) + '.color.png'


def scenename(index):
    return 'frame-' + str(withleading(index)) + '.scene.npy'


# now visualization and saving
def to_uint8(img):
    # normalize and cast to uint8
    img -= img.min()
    img *= 255.0 / np.max(img)
    return np.uint8(np.around(img))


def save(filename, img):
    cv2.imwrite(filename, to_uint8(img))


def save_from_validation(prediction, SAVE_IN_FOLDER, counter, truth = None, epoch = 0, uncertainty = None):
    img = prediction.cpu().numpy()
    img = cv2.resize(img.transpose((1, 2, 0)), dsize=(640, 480))
    # set to zero nan values
    if truth is not None:
        truth = truth.cpu().numpy()
        mask_to_zero = np.isnan(truth)
        truth[mask_to_zero] = 0
        truth_img = cv2.resize(truth.transpose((1, 2, 0)), dsize=(640, 480))
        save(os.path.join(SAVE_IN_FOLDER, 'validation_image_' + str(counter) + '_epoch_' + str(epoch) + '_truth.png'), truth_img)
    if uncertainty is not None:
        u_img = uncertainty.cpu().numpy()
        u_img = cv2.resize(u_img.transpose((1, 2, 0)), dsize=(640, 480))
        # set to zero nan values
        save(os.path.join(SAVE_IN_FOLDER, 'validation_image_' + str(counter) + '_epoch_' + str(epoch) +'_uncertainty.png'), u_img)

    save(os.path.join(SAVE_IN_FOLDER, 'validation_image_' + str(counter) +'_epoch_' + str(epoch) + '_prediction.png'), img)


def save_uncertainty(prediction, SAVE_IN_FOLDER, counter, epoch = 0):
    img = prediction.cpu().numpy()
    img = cv2.resize(img.transpose((1, 2, 0)), dsize=(640, 480))
    # set to zero nan values
    save(os.path.join(SAVE_IN_FOLDER, 'test_image_' + str(counter) + 'u' + '.png'), img)


def save_from_testing_with_params(prediction, SAVE_IN_FOLDER, counter, args):
    #prediction to image
    img = prediction.cpu().numpy()
    img = cv2.resize(img.transpose((1, 2, 0)), dsize=(640, 480))

    #add a band to put args in
    band_height = 30*len(args) + 10
    img_tot = np.zeros((480 + band_height, 640, 3))
    img_tot[band_height:, :, :] = img
    img_tot = to_uint8(img_tot)

    for index, key in enumerate(args):
        text = key + ': ' + str(args[key])
        img_tot = cv2.putText(img_tot, text, (5, (index+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    cv2.imwrite(os.path.join(SAVE_IN_FOLDER, 'test_image' + str(counter) + '.png'), img_tot)


def msavetxt(mlist,PATH):
    np.savetxt(PATH, np.array(mlist))



class ProgressBar:
    def __init__(self, task_name, task_total_iterations, complete_message='task completed', decimal_places=0, length=50,
                 fill='\u2588', step=1):
        self.prefix = task_name
        self.total = task_total_iterations
        self.length = length
        self.decimals = decimal_places
        self.fill = fill
        self.complete_message = complete_message
        self.step = (step / 100) * task_total_iterations
        self.high = 0
        self.start = time.time()
        self.flag = True

    def __call__(s, iteration, message=''):
        iteration += 1
        if s.flag:
            # first time it is called
            print('\n')
            s.flag = False
            s.start = time.time()

        elif iteration > s.high:
            s.high += s.step
            P = (iteration / float(s.total))
            time_remaining = round((time.time() - s.start) * (1 / P - 1))
            # strings
            percent = ("{0:." + str(s.decimals) + "f}").format(100 * P)

            filledLength = int(s.length * iteration // s.total)
            bar = s.fill * filledLength + '-' * (s.length - filledLength)

            time_info = 'time remaining: ' + str(datetime.timedelta(seconds=time_remaining))
            #print('\r%s |%s| %s%% ' % (s.prefix, bar, percent) + '    ' + time_info + '    ' + message, end = '\r')
            sys.stdout.write('\r%s |%s| %s%% ' % (s.prefix, bar, percent) + '    ' + time_info + '    ' + message)

        if iteration == s.total:
            # Print on Complete
            if s.step > 1:
                bar = s.fill * int(s.length * iteration // s.total)
                print('\r%s |%s| %s%% ' % (s.prefix, bar, ("{0:." + str(s.decimals) + "f}").format(100)), end='\r')
            print('\n' + s.complete_message)
            print('task took ' + str(datetime.timedelta(seconds=(int(time.time() - s.start)))))
            print('\n')




class SceneGroundTruthGenerator:
    def __init__(self, **params):
        #I need the data path
        #width and height of original image
        #width and height of output data
        #datanum
        #focus
        '''
        params = {'data_path': 10, 'input_width': 10,'input_height': 10, 'out_width':10, 'out_height':10,
                  'data_num': 10, 'fx':585, 'fy':585}
        '''
        self.data_path = params['data_path']
        self.width, self.height = params['input_width'], params['input_height']
        self.OUT_WIDTH, self.OUT_HEIGHT = params['out_width'], params['out_height']
        self.datanum = params['data_num']
        self.fx, self.fy = params['fx'], params['fy']
        self.refresh = params['refresh']

    def __call__(self, sequences, save_images = False):
        sequences_train = sequences
        OUT_WIDTH = self.OUT_WIDTH
        OUT_HEIGHT = self.OUT_HEIGHT
        # data path
        data_path = self.data_path
        # number of images per sequence
        datanum = self.datanum

        # define static stuff
        # will be moved shortly
        width = self.width
        height = self.height
        u = np.array([[i for i in range(width)] for j in range(height)])
        v = np.array([[j for i in range(width)] for j in range(height)])
        [u0, v0] = [width // 2, height // 2]
        [fx, fy] = [self.fx, self.fy]
        u = (u - u0) / fx
        v = (v - v0) / fy

        '''REFORM DATASET TO OUR LIKING'''
        for sequence in sequences_train:
            # get interesting folders
            [rgb_path, poses_path, depth_path, scene_path] = getPaths(data_path, sequence)
            print('getting scene coordinates for ' + sequence)

            if (os.path.exists(os.path.join(scene_path, scenename(1))) and (not self.refresh)):
                print('SKIPPING: files already exist for this sequence')
                continue
            #progress_bar = ProgressBar('generating', datanum)
            for i in range(datanum):
                #progress_bar(i)
                # load depth image
                z = cv2.imread(os.path.join(depth_path, depthname(sequence, i)), cv2.IMREAD_ANYDEPTH).astype(float)
                # set depth lower than 1 cm to None
                mask_to_none = z < 1
                z[mask_to_none] = None

                # load the transformation matrix from the txt file
                T_cam2world = np.loadtxt(os.path.join(poses_path, posename(i)))
                x = u * z
                y = v * z

                # get scene coordinates
                xw = x * T_cam2world[0, 0] + y * T_cam2world[0, 1] + z * T_cam2world[0, 2] + T_cam2world[0, 3]
                yw = x * T_cam2world[1, 0] + y * T_cam2world[1, 1] + z * T_cam2world[1, 2] + T_cam2world[1, 3]
                zw = x * T_cam2world[2, 0] + y * T_cam2world[2, 1] + z * T_cam2world[2, 2] + T_cam2world[2, 3]

                # img = desired output of the network (i.e. label for 1st training step)
                img_scene_coords = np.stack((xw, yw, zw), 2)

                # img_scene_coords = cv2.resize(img_scene_coords, dsize=(OUT_WIDTH, OUT_HEIGHT),
                #                               interpolation=cv2.INTER_CUBIC)
                #this is their actual implementation
                img_scene_coords = img_scene_coords[::8, ::8]

                # save scene coord array to a .npy file to be used later for training
                np.save(os.path.join(scene_path, scenename(i)), img_scene_coords)

                # optional: save as image with cool colors - OSS: due to normalization loses all information
                if save_images:
                    mask_to_zero = np.isnan(img_scene_coords)
                    img_scene_coords[mask_to_zero] = 0
                    save(os.path.join(scene_path, scenename(i)) + '.png', img_scene_coords)


class Logger:
    def __init__(self):
        self.data = {}

    def __call__(self, keys_and_values):
        for key, value in keys_and_values.items():
            if(key in self.data):
                self.data[key].append(value)
            else:
                self.data[key] = [value]

    def get(self, key):
        return np.array(self.data[key])

    def call(self, key, numpy_function):
        # call a numpy function on the array memorized at key
        vector = np.array(self.data[key])
        return numpy_function(vector)

    def clear(self, keys = None):
        # clear data in keys
        if keys is None:
            self.data = {}
        else:
            for key in keys:
                self.data[key] = []

    def save_plot(self, keys, graph_titles = None, folder = os.getcwd(), smooth = None):
        # save a plot of keys, you can apply a gaussian smoothing to it, default is gaussian
        # with variance 30
        plt.style.use('fivethirtyeight')
        if smooth is None:
            smooth = [30]
        if graph_titles is None:
            graph_titles = keys

        for idx, key in enumerate(keys):
            vector = np.array(self.data[key])
            indexes = np.array([i for i in range(max(vector.shape))])

            vector = gaussian_filter1d(vector, smooth[idx])

            plt.plot(indexes, vector)
            plt.title(graph_titles[idx])

            plt.savefig(os.path.join(folder, graph_titles[idx] + '.png'))
            plt.clf()