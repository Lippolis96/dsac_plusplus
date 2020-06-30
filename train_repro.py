#!/usr/bin/env python
import os
import numpy as np
import torch
import torch.optim as optim

import utils.train_repro_util as repro_util
from models.model import NetVanilla as Net
import utils.utils_global as utils_global
from utils.utils_global import ProgressBar
from properties.properties_repro import PropertiesRepro
from properties.properties_global import PropertiesGlobal

properties_global = PropertiesGlobal()
properties_repro = PropertiesRepro()


training_generator, validation_generator = properties_repro.get_dataloaders()
# declare net and loss
net = Net()
# load model
net.load_state_dict(torch.load(properties_repro.path_model_to_load, map_location=properties_repro.device))
# send net to cuda
net.to(properties_repro.device)


# define optimizer
optimizer = optim.Adam(net.parameters(), lr=properties_repro.lr)
# define learning rate scheduling
milestones = np.array([25, 37, 50, 62])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)


test_loss = float('Inf')
for epoch in range(properties_repro.n_epochs):
    print('STARTED EPOCH: ', epoch)
    # Training
    running_loss = 0
    counter = 0
    mean_u = 0
    if not properties_repro.cluster: progress_indicator = ProgressBar('training', int(len(training_generator)))

    for idx, (local_batch, local_labels, pose) in enumerate(training_generator):
        if not properties_repro.cluster and idx > 0: progress_indicator(idx, message='running loss: ' + utils_global.digits(running_loss / idx) +
                                                                        '    running mean uncertainty: ' + utils_global.digits(max([mean_u/idx, 0])))

        optimizer.zero_grad()
        # Transfer to GPU
        local_batch, local_labels, pose = local_batch.to(device=properties_repro.device, dtype=torch.float), local_labels.to(
            device=properties_repro.device,
            dtype=torch.float), pose.squeeze().to(device=properties_repro.device, dtype=torch.float)

        # forward
        prediction, uncertainty = net(local_batch)
        # compute loss only for pixels for which we know ground truth
        if properties_repro.mask_black_pixels:
            mask_to_zero = torch.isnan(local_labels).to(properties_repro.device)
        else:
            #this is a fake mask
            mask_to_zero = torch.isnan(prediction).to(properties_repro.device)
        # compute repro loss
        loss_u, loss_classic = repro_util.project_mask_gaussian(prediction, pose, properties_global.camera_matrix_torch, properties_global.true_pixel_coords[0],
                                                                properties_global.true_pixel_coords[1], properties_repro.pixel_error_threshold, mask_to_zero,
                                                                uncertainty)
        # can choose whether to train on uncertain loss or on standard reprojection error
        if properties_repro.with_uncertainty:
            loss = loss_u
            mean_u += float(torch.mean(uncertainty))
        else:
            loss = loss_classic
            mean_u += float(-1)
        loss.backward()
        running_loss += float(loss)

        if len(properties_repro.sequences_validation) == 0 and properties_repro.save_outputs and counter < 3:
            counter += 1
            # save images for debugging
            utils_global.save_from_validation(prediction[0].clone().detach(), properties_repro.save_in_folder, counter, truth=local_labels[0], epoch=None)

        optimizer.step()
    scheduler.step()
    running_loss = running_loss / len(training_generator)


    print('EPOCH: ', epoch)
    print('repro loss: ', utils_global.digits(running_loss))

    # Validation
    if len(properties_repro.sequences_validation) > 0:
        counter = 0
        if not properties_repro.cluster: progress_indicator = ProgressBar('validating', int(len(validation_generator)))
        with torch.set_grad_enabled(False):
            running_loss_val = 0
            for idx, (local_batch, local_labels, pose) in enumerate(validation_generator):
                if not properties_repro.cluster and idx > 0: progress_indicator(idx, message='running loss: ' + utils_global.digits(running_loss_val / idx))

                # Transfer to GPU
                local_batch, local_labels, pose = local_batch.to(device=properties_repro.device, dtype=torch.float), local_labels.to(
                    device=properties_repro.device, dtype=torch.float), pose.squeeze().to(device=properties_repro.device, dtype=torch.float)
                # forward
                prediction, uncertainty = net(local_batch)
                if properties_repro.mask_black_pixels:
                    mask_to_zero = torch.isnan(local_labels).to(properties_repro.device)
                else:
                    # this is a fake mask
                    mask_to_zero = torch.isnan(prediction).to(properties_repro.device)
                loss_u, loss_classic = repro_util.project_mask_gaussian(prediction, pose, properties_global.camera_matrix_torch, properties_global.true_pixel_coords[0], properties_global.true_pixel_coords[1], properties_repro.pixel_error_threshold, mask_to_zero)
                if properties_repro.with_uncertainty:
                    loss = loss_u
                else:
                    loss = loss_classic
                if properties_repro.save_outputs and counter < 3:
                    counter += 1
                    # save images for debugging
                    utils_global.save_from_validation(prediction[0], properties_repro.save_in_folder, counter, truth=local_labels[0], epoch=None)

                running_loss_val += float(loss)
            running_loss_val = running_loss_val / len(validation_generator)
        print('validation loss: ', utils_global.digits(running_loss_val))
    # save model if test loss is improved
    # change here if you train on classic loss
    if running_loss < test_loss:
        path = os.path.join(properties_repro.save_in_folder, 'repro_loss_model.pth')
        torch.save(net.state_dict(), path)
        test_loss = running_loss
        print('SAVED MODEL AT EPOCH ', epoch)
