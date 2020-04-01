import getopt, sys, os, torch, pathlib, random
import numpy as np
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tools.csv_loader import *
from nn.data_model import Data_Model 
from nn.image_model import Image_Model

def get_data_batch(x, y, i, batch_size):
    # Calc the batch upper limmit
    upper_limit = np.amin([len(x), i*batch_size + batch_size])

    # Get the batch and predictions
    x = x[i*batch_size: upper_limit, :]
    y = y[i*batch_size: upper_limit, :]

    return x, y


def train_data(save_dir, train_x, train_y, val_x, val_y, epochs, batch_size, lr):
    # Set model name
    model_name = 'data'
    # Create the model
    model = Data_Model().double()
    # Verify the model
    print(model)

    # Define the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

     # Store loss and accuracy so we can graph later
    epoch_train_loss = np.zeros(epochs)
    epoch_train_acc = np.zeros(epochs)
    epoch_val_loss = np.zeros(epochs)
    epoch_val_acc = np.zeros(epochs)
    # Start the training loop
    for e in range(epochs):

        # Training loop
        model.train()
        for i in range(int(np.ceil(len(train_x) // batch_size))):

            # Get the training batch
            x, y = get_data_batch(train_x, train_y, i, batch_size)
            # Cast to tensor
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y))

            # Set the gradient buffer to zero
            model.zero_grad()
            # Calc the loss and accumulate the grad
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            # Update the grad
            optimizer.step()

            # Store the loss and the accuracy
            epoch_train_loss[e] += loss 
            epoch_train_acc[e] += (pred == y).sum().item()


        # Validation loop
        model.eval()
        for i in range(len(val_x)):

            # Get the samples
            x = val_x[i, :]
            y = val_y[i, :]
            # Cast to tensor
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y))

            # Calc the loss and accumulate the grad
            pred = model(x)
            loss = criterion(pred, y)

            # Store the loss and acc
            epoch_val_loss[e] += loss
            epoch_val_acc[e] += (pred == y).sum().item()

        # Calc the mean loss
        epoch_val_loss[e] = epoch_val_loss[e] / len(val_x)
        epoch_val_acc[e] = epoch_val_acc[e] / len(val_x)
        epoch_train_loss[e] = epoch_train_loss[e] / len(train_x)
        epoch_train_acc[e] = epoch_train_acc[e] / len(train_x)


        # Print epoch info
        print('Epoch #' + str(e) + '\tTraining loss: ' + str(epoch_train_loss[e]) + ' \tEpoch validation loss: ' + str(epoch_val_loss[e]))

        # Store the model
        torch.save(model.state_dict(), save_dir + '/' + model_name + '_epoch' + str(e) + '.pth')        


    return epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc

    
def train_images(save_dir, train, val, epochs, batch_size, lr):
    # Set model name
    model_name = 'images'
    # Create the model
    model = Image_Model().double()
    # Verify the model
    print(model)
    # Define the loss and optimizer
    criterion = nn.MSELoss()   
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)      
    #optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  
    
    # Store loss and accuracy so we can graph later
    epoch_train_loss = np.zeros(epochs)
    epoch_train_acc = np.zeros(epochs)
    epoch_val_loss = np.zeros(epochs)
    epoch_val_acc = np.zeros(epochs)
    # Start the training loop
    for e in range(epochs):

        # Copy validation and training names
        train_rem =  train[:]
        val_rem =  val[:]

        # Training loop
        model.train()
        for i in range(int(np.ceil(len(train) // batch_size))):
            # Get the training batch
            x, y = load_images(data_path, train_rem, np.amin([batch_size, len(train_rem)]))
            # Normalize the images and re arrange dimensions
            x = np.transpose(x, (0, 3, 2, 1)) / 255 
            y, _ = minmax_normmalization(pd.DataFrame(y))
            # Cast to tensor
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y.values))

            # Set the gradient buffer to zero
            model.zero_grad()
            # Calc the loss and accumulate the grad
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            # Update the grad
            optimizer.step()

            # Store the loss and the accuracy
            epoch_train_loss[e] += loss 
            epoch_train_acc[e] += (pred == y).sum().item()


        # Validation loop
        model.eval()
        for i in range(len(val)):

            # Get the samples
            x, y = load_images(data_path, val_rem, 1)
            # Normalize the images and re arrange dimensions
            x = np.transpose(x, (0, 3, 2, 1)) / 255 
            y, _ = minmax_normmalization(pd.DataFrame(y))
            # Cast to tensor
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y.values))

            # Calc the loss and accumulate the grad
            pred = model(x)
            loss = criterion(pred, y)

            # Store the loss and acc
            epoch_val_loss[e] += loss
            epoch_val_acc[e] += (pred == y).sum().item()

        # Calc the mean loss
        epoch_val_loss[e] = epoch_val_loss[e] / len(val)
        epoch_val_acc[e] = epoch_val_acc[e] / len(val)
        epoch_train_loss[e] = epoch_train_loss[e] / len(train)
        epoch_train_acc[e] = epoch_train_acc[e] / len(train)


        # Print epoch info
        print('Epoch #' + str(e) + '\tTraining loss: ' + str(epoch_train_loss[e]) + ' \tEpoch validation loss: ' + str(epoch_val_loss[e]))

        # Store the model
        torch.save(model.state_dict(), save_dir + '/' + model_name + '_epoch' + str(e) + '.pth')        


    return epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc



def train(model_name, action, data_path):

    # Start training and backprop
    epochs = int(input('Ingrese el número de epochs que desea entrenar: '))
    batch_size = int(input('Ingrese el batch size: '))
    lr = float(input('Ingrese el lr: '))

    # Create the folder
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")
    hiperparameters = '_' + model_name + '_e' + str(epochs) + '_bs' + str(batch_size) + '_lr' + str(lr)
    save_dir = str(pathlib.Path().absolute()) + '/checkpoints/' + now + hiperparameters
    os.mkdir(save_dir)
    
    # Instantiate the selected model
    if(model_name == 'data'):
        # Load the data and normalize it
        y, x = get_mean_values_dataset(data_path)
        y, y_norm = minmax_normmalization(y)
        x, x_norm = minmax_normmalization(x)
        # Divide the dataset in 70-30 for training and validation
        lim = int(len(y) * 0.7)
        train_x = x.iloc[:lim, :].values
        train_y = y.iloc[:lim, :].values
        val_x = x.iloc[lim:, :].values
        val_y = y.iloc[lim:, :].values
        # Do the training
        epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc = train_data(save_dir, train_x, train_y, val_x, val_y, epochs, batch_size, lr)

    elif(model_name == 'image'):
        # Read all available images in the folder and select the valid ones
        available_images = os.listdir(data_path)
        available_images = select_valid_images(available_images)
        # Shuffle the images
        random.shuffle(available_images)
        # Separate the images into train and validation
        lim = int(len(available_images) * 0.7)

        # Train the model
        epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc = train_images(save_dir, available_images[:lim], available_images[lim:], epochs, batch_size, lr)


    

    # Draw the loss
    plt.plot(np.arange(epochs), epoch_train_loss, 'r-', label='Training loss')
    plt.plot(np.arange(epochs), epoch_val_loss, 'b-', label='Validation loss')
    plt.plot(np.arange(epochs), epoch_train_acc, 'm-', label='Training acc')
    plt.plot(np.arange(epochs), epoch_val_acc, 'c-', label='Validation acc')
    plt.title(model_name + hiperparameters + ' model loss and accuracy')
    plt.ylabel('MSE loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(save_dir + '/' + model_name + hiperparameters + '_loss.png')
    plt.show()

    print('Saving data in: ' + save_dir)







if __name__ == "__main__":
    
    model = ''
    action = ''
    data_path = ''

    try:
        # We want to recognize m, a, d as options with argument thats why 
        # the : follows them, h doesnt need arguments so its alone
        opts, args = getopt.getopt(sys.argv[1:],"hm:a:d:",["model=","action=","data="])
    except getopt.GetoptError:
        print('python3 train.py -m <model> -a <action> -d <data path> \nInference looks for the latest folder with the model name, training creates this folder and saves the weigths there')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python3 train.py -m <model> -a <action> -d <data path> \nInference looks for the latest folder with the model name, training creates this folder and saves the weigths there')
            sys.exit()

        elif opt in ("-m", "--model"):
            if((arg == 'data') or (arg == 'image')):
                model = arg
            else:
                print('Unavailable model, please choose data or image')
                sys.exit()

        elif opt in ("-a", "--action"):
            action = arg
            
        elif opt in ("-d", "--data"):
            data_path = arg


    print('Selected ' + model + ' model for ' + action + ', using the data stored in ' + data_path)
    

    if (action == 'train'):
        train(model, action, data_path)
    
   


