from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import h5py

class ShipsDataset(torch.utils.data.Dataset):

    def __init__(self, in_file):
        super(ShipsDataset, self).__init__()

        self.file = h5py.File(in_file, 'r')
        self.n_images, self.nx, self.ny = self.file['images'].shape

    def __getitem__(self, index):
        input = self.file['images'][index,:,:]
        return input.astype('float32')

    def __len__(self):
        return self.n_images

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

class ResNetWithAdditions(torch.nn.Module):
    def __init__(self, pretrained_model, output_classes):
        super(ResNetWithAdditions, self).__init__()
        self.pretrained_model = pretrained_model
        set_parameter_requires_grad(self.pretrained_model)
        num_ftrs = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(num_ftrs, output_classes)
        #self.first_rlu_layer = nn.ReLU()
        #self.almost_last = nn.Linear(num_ftrs, num_ftrs)
        #self.almost_last_rlu_layer = nn.ReLU()
        #self.fc = nn.Linear(num_ftrs, output_classes)

    def forward(self, inputs):
        return self.pretrained_model.forward(inputs)


def main(use_model):
    data_dir = "./data_ships/"

    # Number of classes in the dataset
    num_classes = 26

    # Batch size for training (change depending on how much memory you have)
    batch_size = 200

    # Number of epochs to train for
    num_epochs = 3

    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                torch.cuda.empty_cache()
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                i = 0
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    print(i)
                    i+= 1
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'test':
                    val_acc_history.append(epoch_acc)
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    if not use_model :
        model_ft = models.resnet101(pretrained=True)
        model_ft = ResNetWithAdditions(model_ft, num_classes)
    else :
        model_ft = torch.load('models/resnet_ships_try_with_rlu')
    input_size = 300
    
    for i in range(50) :
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}

        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=16, pin_memory = True) for x in ['train', 'test']}
        # Create training and validation dataloaders
        #trainset = torchvision.datasets.CIFAR10(root='./data_CIFAR', train=True,
                                        #download=True, transform=data_transforms['train'])
        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                #shuffle=True, num_workers=8)

        #testset = torchvision.datasets.CIFAR10(root='./data_CIFAR', train=False,
                                            #download=True, transform=data_transforms['val'])
        #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                #shuffle=False, num_workers=8)

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        # params_to_update = model_ft.parameters()
        # print("Params to learn:")
        # params_to_update = []
        # for name,param in model_ft.named_parameters():
        #     if param.requires_grad == True:
        #         params_to_update.append(param)
        #         print("\t",name)

        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

        torch.save(model_ft, 'models/resnet_ships_try_with_rlu')

if __name__ == '__main__':
    main(True)