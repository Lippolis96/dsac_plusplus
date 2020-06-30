import torch
from torch.utils import data
import cv2 as cv2
import numpy as np
from random import choice


def shift_image(img, x, y):
    img = np.roll(img, (y, x, 0), axis=(0, 1, 2))
    img[:y, :, :] = img[y, :, :]
    img = img.transpose((1, 0, 2))
    img[:x, :, :] = img[x, :, :]
    img = img.transpose((1, 0, 2))
    return img


class Dataset_obj(data.Dataset):
    def __init__(self, list_input_locations, list_label_locations, shift = True):
        self.labels = list_label_locations #xyz image (Height, Width, 3Dcoords)
        self.inputs = list_input_locations #rgb imagen (Height, Width, Color)
        self.shift = [0, 1, 2, 3, 4, 5, 6, 7]
        self.indexes = range(len(self.labels))
        self.do_shift = shift

    def __len__(self):
        # returns total number of samples
        return len(self.labels)

    def __getitem__(self, index):
        #if we want to run training in a more stochastic way like they do:
        #index = choice(self.indexes)
        # Returns one input - output pair
        img = cv2.imread(self.inputs[index], cv2.IMREAD_UNCHANGED).astype(float)
        label = np.load(self.labels[index])

        if self.do_shift:
            #pick vertical and horizontal shifts ar random
            shift_x, shift_y = choice(self.shift), choice(self.shift)
            #shift the image, (not the label)
            img = shift_image(img, shift_x, shift_y)
            #if you wanna see that it actually works jusy uncomment this
            #imshow(img)

        # transpose axes since
        # numpy image: Height x Width x Color
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        # I also transpose this... I imagine we will have three channels in input and three in output
        label = label.transpose((2, 0, 1))

        # we have to return X, y
        return torch.from_numpy(img), torch.from_numpy(label)



class Dataset_repro(data.Dataset):
    def __init__(self, list_input_locations, list_label_locations, list_pose_locations, shift = True):

        self.labels = list_label_locations #xyz image (Height, Width, 3Dcoords)
        self.inputs = list_input_locations #rgb imagen (Height, Width, Color)
        self.shift = [0, 1, 2, 3, 4, 5, 6, 7]
        self.pose = list_pose_locations
        self.do_shift = shift

    def __len__(self):
        # returns total number of samples
        return len(self.labels)

    def __getitem__(self, index):
        # Returns one input - output pair
        img = cv2.imread(self.inputs[index], cv2.IMREAD_UNCHANGED).astype(float)
        label = np.load(self.labels[index])

        if self.do_shift:
            #pick vertical and horizontal shifts ar random
            shift_x, shift_y = choice(self.shift), choice(self.shift)
            #shift the image, (not the label)
            img = shift_image(img, shift_x, shift_y)
            #if you wanna see that it actually works jusy uncomment this
            #imshow(img)

        # transpose axes since
        # numpy image: Height x Width x Color
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        # I also transpose this... I imagine we will have three channels in input and three in output
        label = label.transpose((2, 0, 1))

        # get pose
        file = open(self.pose[index], "r")
        pose = [line.split() for line in file]
        pose = np.squeeze(pose)

        # we have to return X, y, pose
        return torch.from_numpy(img), torch.from_numpy(label), pose.astype('float32')


class Dataset_dsac(data.Dataset):
    def __init__(self, list_input_locations, list_label_locations, list_scene_locations=None):
        self.rgbs = list_input_locations  # rgb image (Height, Width, Color)
        self.poses = list_label_locations  # pose file
        self.scenes = list_scene_locations

    def __len__(self):
        # returns total number of samples
        return len(self.poses)

    def __getitem__(self, index):
        # Returns one input - output pair
        img = cv2.imread(self.rgbs[index], cv2.IMREAD_UNCHANGED).astype(float)
        pose = np.loadtxt(self.poses[index])
        if self.scenes is not None:
            scene = np.load(self.scenes[index])
            return torch.from_numpy(img.transpose((2, 0, 1))), pose , torch.from_numpy(scene.transpose((2, 0, 1)))

        return torch.from_numpy(img.transpose((2, 0, 1))), pose