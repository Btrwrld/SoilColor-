import torch, csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 

from torch.autograd import Variable
from tools.training_manager import get_data_batch, batch_accuracy, classify, plot_results



class Data_Model(nn.Module):


    def __init__(self, loss):
        super(Data_Model, self).__init__()

        # Set model name
        self.model_name = 'data'
        self.save_dir = ''
        self.hiperparameters = ''

        # Define loss and optimizer
        self.loss = loss
        self.optimizer = None

        # Input is a feature vector of 1x12 and we want a 1x60 output vector
        self.hidden = nn.Linear(in_features=12, out_features=60)
        self.sig1 = nn.Sigmoid()
        # And we want a 1x3 output so we can make the regression 
        self.output = nn.Linear(in_features=60, out_features=3)
        self.sig2 = nn.Sigmoid()


    def forward(self, x):

        # Since we are trying to imitate the matlab Feedforward Neural Network
        # we'll use a sigmoid activation function here
        x = self.hidden(x)
        x = self.sig1(x)
        # Then a sigmoid in the output
        x = self.output(x)
        x = self.sig2(x)

        return x   

    
    def start_training(self, save_dir, train_x, train_y, val_x, val_y, epochs, batch_size, lr):

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
                x = Variable(torch.from_numpy(x))
                y = Variable(torch.from_numpy(y))

                # According to pytorch documentation:
                # Some optimization algorithms such as Conjugate Gradient and LBFGS 
                # need to reevaluate the function multiple times, so you have to pass 
                # in a closure that allows them to recompute your model. The closure 
                # should clear the gradients, compute the loss, and return it.

                # Add the closure function to calculate an iterative gradient
                def closure():
                    if torch.is_grad_enabled():
                        self.optimizer.zero_grad()
                    output = self(x)
                    loss = self.loss(output, y)
                    if loss.requires_grad:
                        loss.backward()
                    return loss
                self.optimizer.step(closure)

                # Calc loss again to check progress
                pred = self(x)
                loss = closure()

                # Store the loss and the accuracy
                epoch_train_loss[e] += loss 
                epoch_train_acc[e, :] += batch_accuracy(pred, y)


            # Validation loop
            self.eval()
            for i in range(len(val_x)):

                # Get the samples
                x = val_x[i, :]
                y = val_y[i, :]
                # Cast to tensor
                x = Variable(torch.from_numpy(x))
                y = Variable(torch.from_numpy(y))

                # Calc the loss and accumulate the grad
                pred = self(x)
                loss = self.loss(pred, y)

                # Store the loss and acc
                epoch_val_loss[e] += loss
                epoch_val_acc[e, :] += np.isclose(np.array(classify(pred.data.numpy())), y.data.numpy())
                
            
            # Calc the mean loss
            epoch_val_loss[e] = epoch_val_loss[e] / len(val_x)
            epoch_val_acc[e, :] = epoch_val_acc[e, :] / len(val_x)
            epoch_train_loss[e] = epoch_train_loss[e] / num_batches
            epoch_train_acc[e, :] = epoch_train_acc[e, :] / len(train_x)


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


    def start_test(self, test_x, test_y, log_path):

        l = len(test_x)

        epoch_test_loss = np.zeros(l)
        epoch_test_acc = np.zeros([l, 3])

        # Validation loop
        self.eval()
        for i in range(len(test_x)):

            # Get the samples
            x = test_x[i, :]
            y = test_y[i, :]
            # Cast to tensor
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y))

            # Calc the loss and accumulate the grad
            pred = self(x)
            loss = self.loss(pred, y)

            # Store the loss and acc
            epoch_test_loss[i] += loss
            epoch_test_acc[i, :] += np.isclose(np.array(classify(pred.data.numpy())), y.data.numpy())

        mean_loss = np.mean(epoch_test_loss)
        h_acc = np.mean(epoch_test_acc[:, 0])
        c_acc = np.mean(epoch_test_acc[:, 1])
        v_acc = np.mean(epoch_test_acc[:, 2])
        accuracy = np.sum(epoch_test_acc, 1)
        accuracy[accuracy < 3] = 0
        accuracy[accuracy == 3] = 1
        accuracy = np.mean(accuracy)

        # Write csv     
        logs = open('data_test_results.csv', 'w')
        with logs:
            writer = csv.writer(logs)

            # Write the header
            writer.writerow(["Mean_loss", "Accuracy", "Hue_accuracy", "Chroma_accuracy", "Value_accuracy"])
            # Write the data
            writer.writerow([mean_loss, accuracy, accuracy, h_acc, c_acc, v_acc])

        print("Mean_loss: " + str(mean_loss))
        print("Accuracy: " + str(accuracy))
        print("Hue_accuracy: " + str(h_acc))
        print("Chroma_accuracy: " + str(c_acc))
        print("Value_accuracy: " + str(v_acc))
#python3 train.py -m data -a train -d /home/erick/google_drive/PARMA/SoilColor/Images/o_marked_definitive/

