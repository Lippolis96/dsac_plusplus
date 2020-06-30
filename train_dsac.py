#!/usr/bin/env python
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils.utils_global as utils_global
from models.model import NetVanilla as Net
import utils.train_dsac_util as ransac_util
import utils.train_dsac_sub as sub
from utils.utils_global import ProgressBar

from properties.properties_dsac import PropertiesDsac
from properties.properties_global import PropertiesGlobal

properties_global = PropertiesGlobal()
properties_dsac = PropertiesDsac()

# Loading pre-trained model
model = Net()
model.load_state_dict(torch.load(properties_dsac.model_path, map_location=properties_global.device))
model.to(properties_global.device)


training_generator, validation_generator = properties_dsac.get_dataloaders()
optimizer = optim.Adam(model.parameters(), lr=properties_dsac.learning_rate)

print('INFO TRAINING: ')
print('repro error threshold: {} pixels'.format(properties_dsac.inlier_threshold))
print('number of refinement iters: {}'.format(properties_dsac.number_refinement_iterations))

if properties_dsac.starting_epoch == 0:
    losses_log = np.zeros((properties_dsac.max_epochs, 3))
else:
    losses_old = np.loadtxt(os.path.join(properties_dsac.save_in_folder, 'log ransac.txt'))
    losses_log = np.zeros((properties_dsac.max_epochs, 3))
    losses_log[:properties_dsac.starting_epoch,:] = losses_old[:properties_dsac.starting_epoch,:]
    print('previous log file loaded')

best_train_loss = float('Inf')
for epoch in range(properties_dsac.starting_epoch, properties_dsac.max_epochs):
    running_training_loss = 0
    counter = 0

    if not properties_dsac.cluster: pb = ProgressBar('training epoch: ' + str(epoch), len(training_generator), step=0.05, decimal_places=2)

    #TRAINING LOOP
    for index, (local_batch, pose_label) in enumerate(training_generator):
        if not properties_dsac.cluster and index > 0: pb(index, message='running loss: ' + '{:.2f}'.format(running_training_loss / (index)))

        optimizer.zero_grad()
        #get scene coordinate prediction and calculate L1 loss (for debuggind purposes)
        local_batch = local_batch.to(device=properties_global.device, dtype=torch.float)
        prediction, uncertainty = model(local_batch)
        if not properties_dsac.with_uncertainty:
            uncertainty = torch.ones(properties_global.height_out*properties_global.width_out).unsqueeze(0)
        prediction = prediction.squeeze(0)
        uncertainty = uncertainty.squeeze(0)

        #get ground truths
        R_cam_to_world_true, t_cam_to_world_true = ransac_util.get_truth(pose_label)
        #get losses and scores (soft inlier counts) for 256 hypotheses
        # if you want to train without uncertainty flag_sampling='uniform'
        losses, scores, _ = sub.get_losses_and_scores(prediction, uncertainty, properties_dsac.number_hypotheses, R_cam_to_world_true, t_cam_to_world_true)

        #get top n hypotheses
        scores, mask_top = torch.topk(scores, properties_dsac.num_top)

        losses = losses[mask_top]
        #deal with the probability ditribution
        probabilities = F.softmax(scores, dim=0)
        expected_loss = torch.dot(probabilities, losses)

        expected_loss.backward()

        #check for nans in gradient and skip step if you find any
        if(torch.isnan(model.conv11.bias.grad).any()):
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), properties_dsac.clipping_value)

        optimizer.step()
        running_training_loss += float(expected_loss)

        if len(properties_dsac.sequences_validation) == 0 and properties_dsac.save_outputs and counter < 3:
            counter += 1
            # save images for debugging
            utils_global.save_from_validation(prediction.clone().detach(), properties_dsac.save_in_folder, counter, epoch=epoch)

    running_training_loss /= len(training_generator)
    print('\npose training loss: ', utils_global.digits(running_training_loss))
    # END TRAINING PART

    # Validation
    if len(properties_dsac.sequences_validation) > 0:
        counter = 0
        if not properties_dsac.cluster: pb = ProgressBar('validating epoch: ' + str(epoch), len(validation_generator), step=0.05, decimal_places=2)
        with torch.set_grad_enabled(False):
            running_loss_val = 0
            for index, (local_batch, pose_label) in enumerate(training_generator):
                if not properties_dsac.cluster and index > 0: pb(index, message='running loss: ' + '{:.2f}'.format(running_training_loss / (index)))
                # get scene coordinate prediction and calculate L1 loss (for debuggind purposes)
                local_batch = local_batch.to(device=properties_global.device, dtype=torch.float)
                prediction, uncertainty = model(local_batch)
                prediction = prediction.squeeze(0)
                uncertainty = uncertainty.squeeze(0)
                # get ground truths
                R_cam_to_world_true, t_cam_to_world_true = ransac_util.get_truth(pose_label)
                # get losses and scores (soft inlier counts) for 256 hypotheses
                losses, scores, _ = sub.get_losses_and_scores(prediction, uncertainty, properties_dsac.number_hypotheses, R_cam_to_world_true, t_cam_to_world_true, flag_sampling='proportional')

                # get top n hypotheses
                scores, mask_top = torch.topk(scores, properties_dsac.num_top)

                losses = losses[mask_top]
                # deal with the probability ditribution
                probabilities = F.softmax(scores, dim=0)
                expected_loss = torch.dot(probabilities, losses)

                running_loss_val += float(expected_loss)

                if properties_dsac.save_outputs and counter < 3:
                    counter += 1
                    # save images for debugging
                    utils_global.save_from_validation(prediction.clone().detach(), properties_dsac.save_in_folder, counter, epoch=epoch)

        running_training_loss /= len(validation_generator)
        print('pose validation loss: ', utils_global.digits(running_loss_val))


    # save model if train loss is improved
    if running_training_loss < best_train_loss or properties_dsac.save_all_models:
        PATH = os.path.join(properties_dsac.save_in_folder, 'dsac_' + properties_dsac.model_name + '.pth')
        torch.save(model.state_dict(), PATH)
        best_train_loss = running_training_loss
        print('SAVED MODEL AT EPOCH ', epoch)

    if properties_dsac.text_log:
        losses_log[epoch, 0] = epoch
        losses_log[epoch, 1] = running_training_loss
        if len(properties_dsac.sequences_validation) > 0:
            losses_log[epoch, 2] = running_loss_val
        np.savetxt(os.path.join(properties_dsac.save_in_folder, 'log ransac.txt'), losses_log, fmt='%.2f')


