import getopt, sys, os, torch, pathlib, random
import numpy as np
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from tools.data_loader import *
from nn.image_model import Image_Model
from nn.data_modelv2 import Data_Model 





def plot_results(save_dir, model_name, epochs, hiperparameters, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc):

    # Create the image
    fig, axs = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0.5})
    fig.suptitle(hiperparameters + ' model loss and accuracy')

    # Plot the loss
    axs[0].set_title('Model loss')
    axs[0].plot(np.arange(epochs), epoch_train_loss, 'r-', label='Training')
    axs[0].plot(np.arange(epochs), epoch_val_loss, 'b-', label='Validation')
    axs.flat[0].set(xlabel='Epochs', ylabel='MSE loss')
    axs[0].legend()

    # Plot the training accuracy
    axs[1].set_title('Training accuracy')
    axs[1].plot(np.arange(epochs), epoch_train_acc[:, 0], 'g-', label='Hue')
    axs[1].plot(np.arange(epochs), epoch_train_acc[:, 1], 'c-', label='Chroma')
    axs[1].plot(np.arange(epochs), epoch_train_acc[:, 2], 'y-', label='Value')
    axs.flat[1].set(xlabel='Epochs', ylabel='Accuracy')
    axs[1].legend()

    # Plot the validation accuracy
    axs[2].set_title('Validation accuracy')
    axs[2].plot(np.arange(epochs), epoch_val_acc[:, 0], 'g-', label='Hue')
    axs[2].plot(np.arange(epochs), epoch_val_acc[:, 1], 'c-', label='Chroma')
    axs[2].plot(np.arange(epochs), epoch_val_acc[:, 2], 'y-', label='Value')
    axs.flat[2].set(xlabel='Epochs', ylabel='Accuracy')
    axs[2].legend()

    plt.savefig(save_dir + '/' + model_name + hiperparameters + '_loss.png')
    plt.show()

    print('Saving data in: ' + save_dir)



#   Function designed to be a wrapper around the networks training, it
#   stores the weights in a folder named after the parameters used and
#   the model name, also generates a plot with the training results
#
#   Parameters:
#       model: Variable that chooses wich model to train
#       data_path: Directory containing all the data necesary to train the model
#
#   Return: None
def train(model_name, data_path):

    # Start training and backprop
    epochs = int(input('Ingrese el número de epochs que desea entrenar: '))
    batch_size = int(input('Ingrese el batch size: '))
    lr = float(input('Ingrese el lr: '))
    
    # Instantiate the selected model
    if(model_name == 'data'):
        # Load the data and normalize it
        x, y = get_mean_values_dataset(data_path)
        y, y_norm = minmax_normmalization(y)
        x, x_norm = minmax_normmalization(x)
        # Divide the dataset in 70-30 for training and validation
        lim = int(len(y) * 0.7)
        train_x = x.iloc[:lim, :].values
        train_y = y.iloc[:lim, :].values
        val_x = x.iloc[lim:, :].values
        val_y = y.iloc[lim:, :].values


        # Generate model and define the optimizer
        model = Data_Model(loss = nn.MSELoss()).double()
        model.optimizer = optim.LBFGS(model.parameters(), history_size=10, max_iter=10)
        #model.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        #model.optimizer  = optim.SGD(model.parameters(), lr=lr)


    elif(model_name == 'image'):
        # Read all available images in the folder and select the valid ones
        available_images = os.listdir(data_path)
        available_images = select_valid_images(available_images)
        # Shuffle the images
        random.shuffle(available_images)
        # Load the data
        x, y = load_images(data_path, available_images)    
        # Normalize the data and re arrange image dimensions
        x = np.transpose(x, (0, 3, 2, 1)) / 255 
        y, _ = minmax_normmalization(pd.DataFrame(y))
        # Get the numpy array containing the values
        y = y.values
        # Divide the dataset in 70-30 for training and validation
        lim = int(len(y) * 0.7)
        train_x = x[:lim]
        train_y = y[:lim]
        val_x = x[lim:]
        val_y = y[lim:]

        # Create the model and define the optimizer
        model = Image_Model(loss = nn.MSELoss()).double()
        model.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)


    # In case we want to use the whole batch
    if(batch_size < 0):
        batch_size = len(train_x)


    # Create the folder
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")
    hiperparameters = '_' + model_name + '_e' + str(epochs) + '_bs' + str(batch_size) + '_lr' + str(lr)
    save_dir = str(pathlib.Path().absolute()) + '/checkpoints/' + now + hiperparameters
    os.mkdir(save_dir)
    
    # Verify the model
    print(model)
    # Do the training
    epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc = model.start_training(save_dir, train_x, train_y, val_x, val_y, epochs, batch_size, lr)


    
    plot_results(save_dir, model.model_name, epochs, hiperparameters, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc)



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
        train(model, data_path)
    
   
# python3 train.py -m image -a train -d ../Images/o1_fused/
# python3 train.py -m data -a train -d ../Images/o1_marked/

