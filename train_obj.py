#!/usr/bin/env python
import os
import numpy as np
import torch
import torch.optim as optim

import utils.utils_global as utils_global
from utils.utils_global import ProgressBar
from properties.properties_obj import PropertiesObj
from models.model import NetVanilla as Net

properties_obj = PropertiesObj()
training_generator, validation_generator = properties_obj.get_dataloaders()

# declare net and loss
net = Net()
# send net to cuda
net.to(properties_obj.device)

# define optimizer
optimizer = optim.Adam(net.parameters(), lr=properties_obj.lr)
# define learning rate scheduling
milestones = np.array([25, 37, 50, 62])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

test_loss = float('Inf')
# Loop over epochs
for epoch in range(properties_obj.n_epochs):
    # Training
    counter = 0
    running_loss = 0
    if not properties_obj.cluster: pb = ProgressBar('training', int(len(training_generator)))
    for idx, (local_batch, local_labels) in enumerate(training_generator):
        if not properties_obj.cluster and idx > 0: pb(idx, message ='running loss: ' + utils_global.digits(running_loss / idx))
        optimizer.zero_grad()
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device=properties_obj.device, dtype=torch.float), local_labels.to(device=properties_obj.device, dtype=torch.float)
        # forward
        prediction, uncertainty = net(local_batch)
        loss_u, loss_L1 = utils_global.l1_loss_u(prediction, local_labels, uncertainty)
        # you can choose whether to train on uncertain loss or on normal L1 loss
        if properties_obj.with_uncertainty:
            loss = loss_u
        else:
            loss = loss_L1
        loss.backward()
        optimizer.step()

        running_loss += float(loss)

        if len(properties_obj.sequences_validation) == 0 and properties_obj.save_outputs_from_validation and counter < 3:
            counter += 1
            # save images for debugging
            utils_global.save_from_validation(prediction[0].clone().detach(), properties_obj.save_in_folder, counter,
                                              truth=local_labels[0], epoch=None)


    scheduler.step()
    mean_training_loss = running_loss / len(training_generator)
    print('EPOCH: ', epoch)
    print('traning loss: ', utils_global.digits(mean_training_loss))

    # Validation
    if(len(properties_obj.sequences_validation) > 0):
        # print('evaluating validation loss')
        counter = 0
        if not properties_obj.cluster: pb = ProgressBar('validating', int(len(validation_generator)))
        with torch.set_grad_enabled(False):
            running_loss_val = 0
            for idx, (local_batch, local_labels) in enumerate(validation_generator):
                if not properties_obj.cluster and idx > 0: pb(idx, message='running loss: ' + utils_global.digits(running_loss_val / idx))
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device=properties_obj.device, dtype=torch.float), local_labels.to(
                    device=properties_obj.device, dtype=torch.float)

                prediction, uncertainty = net(local_batch)
                loss_u, loss_L1 = utils_global.l1_loss_u(prediction, local_labels, uncertainty)
                if properties_obj.with_uncertainty:
                    loss = loss_u
                else:
                    loss = loss_L1

                if properties_obj.save_outputs_from_validation and counter < 4:
                    counter += 1
                    # save images for debugging
                    prediction = prediction[0]
                    truth = local_labels[0]
                    utils_global.save_from_validation(prediction, properties_obj.save_in_folder, counter, truth = truth, epoch = None, uncertainty=uncertainty.clone().detach())

                running_loss_val += float(loss)
        validation_loss = running_loss_val / len(validation_generator)
        print('validation loss: ', validation_loss)

    if mean_training_loss < test_loss:
        path = os.path.join(properties_obj.save_in_folder, 'l1_loss_model' + '.pth')
        torch.save(net.state_dict(), path)
        print('saving model at epoch ' + str(epoch))
        test_loss = mean_training_loss