import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Variable

from tools.training_manager import get_data_batch, batch_accuracy, classify, plot_results



class Image_Model(nn.Module):
    def __init__(self, loss):
        super(Image_Model, self).__init__()


        # Set model name
        self.model_name = 'images'
        self.save_dir = ''
        self.hiperparameters = ''

        # Define loss and optimizer
        self.loss = loss
        self.optimizer = None

        self.conv1= nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1)
        self.relu1= nn.ELU()
        self.norm1= nn.BatchNorm2d(64)
        self.pool1= nn.MaxPool2d(kernel_size=2, stride=2)
       
        
        self.conv2= nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.relu2= nn.ELU()
        self.norm2= nn.BatchNorm2d(32)
        self.pool2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=2048, out_features=500) 
        self.sigm1 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=500, out_features=3)
        self.sigm2 = nn.Sigmoid() 


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)
                
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.pool2(x)
        
        x = x.view(-1,2048) 

        x = self.fc1(x)
        x = self.sigm1(x)
        x = self.fc2(x)
        x = self.sigm2(x)
        
        return x



    def start_training(self, save_dir, train_x, train_y, val_x, val_y, epochs, batch_size, lr):
       
        # Check if we are using GPU or CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Executing model on: ' + str(device))
        self.to(device)

        # Store loss and accuracy so we can graph later
        epoch_train_loss = np.zeros(epochs)
        epoch_train_acc = np.zeros([epochs, 3])
        epoch_val_loss = np.zeros(epochs)
        epoch_val_acc = np.zeros([epochs, 3])
        best_acc = 0

        # Calc constants
        num_batches = int(np.ceil(len(train_x) // batch_size))
        # Start the training loop
        for e in range(epochs):

            # Training loop
            self.train()
            for i in range(num_batches):
                # Get the training batch
                x, y = get_data_batch(train_x, train_y, i, batch_size)
                # Cast to tensor
                x = Variable(torch.from_numpy(x)).to(device)
                y = Variable(torch.from_numpy(y)).to(device)

                # Set the gradient buffer to zero
                self.zero_grad()
                # Calc the loss and accumulate the grad
                pred = self(x)
                loss = self.loss(pred, y)
                loss.backward()
                # Update the grad
                self.optimizer.step()

                # Check in case we are not in cpu
                if(device != 'cpu'):
                    pred = pred.cpu()
                    y = y.cpu()

                # Store the loss and the accuracy
                epoch_train_loss[e] += loss 
                epoch_train_acc[e, :] += batch_accuracy(pred, y)
 
            # Validation loop
            self.eval()
            for i in range(len(val_x)):

                # Cast to tensor
                x = Variable(torch.from_numpy(np.expand_dims(val_x[i], axis=0))).to(device)
                y = Variable(torch.from_numpy(np.expand_dims(val_y[i], axis=0))).to(device)

                # Calc the loss and accumulate the grad
                pred = self(x)
                loss = self.loss(pred, y)

                # Check in case we are not in cpu
                if(device != 'cpu'):
                    pred = pred.cpu()
                    y = y.cpu()

                # Store the loss and acc
                epoch_val_loss[e] += loss
                epoch_val_acc[e, :] += np.isclose(np.array([classify(pred.data.numpy()[0])]), y.data.numpy())[0]
            
            # Calc the mean loss
            epoch_val_loss[e] = epoch_val_loss[e] / len(val_x)
            epoch_val_acc[e, :] = epoch_val_acc[e] / len(val_x)
            epoch_train_loss[e] = epoch_train_loss[e] / num_batches
            epoch_train_acc[e, :] = epoch_train_acc[e] / len(train_x)


            # Print epoch info
            print('Epoch #' + str(e) + '\tTraining loss: ' + str(epoch_train_loss[e]) + ' \tValidation loss: ' + str(epoch_val_loss[e]))

            # Plot epoch info
            plot_results(self.save_dir, self.model_name, e, self.hiperparameters, epoch_train_loss[:e], epoch_val_loss[:e], epoch_train_acc[:e], epoch_val_acc[:e])

            # Store the model only if the validation is better than before
            epoch_acc = epoch_val_acc[e, :].sum()
            if(epoch_acc >  best_acc):
                best_acc = epoch_acc
                torch.save(self.state_dict(), save_dir + '/' + self.model_name + '.pth')        


        return epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc