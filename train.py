import getopt, sys, os, torch, pathlib
import numpy as np
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tools.data_manager import *
from nn.image_model import Image_Model
from nn.data_model import Data_Model 





#   Function designed to be a wrapper around the networks training, it
#   stores the weights in a folder named after the parameters used and
#   the model name, also generates a plot with the training results
#
#   Parameters:
#       model: Variable that chooses wich model to train
#       data_path: Directory containing all the data necesary to train the model
#
#   Return: None
def train(model_name, data_path, extra=''):

    # Start training and backprop
    epochs = int(input('Ingrese el número de epochs que desea entrenar: '))
    batch_size = int(input('Ingrese el batch size: '))
    lr = float(input('Ingrese el lr: '))
    
    # Instantiate the selected model
    if(model_name == 'data'):
        # Load the training data and normalize it
        train_x, train_y = get_mean_values_dataset(data_path + 'train/', shuffle=False)
        train_x, norm_x = minmax_normmalization(train_x)
        train_y = normalize_targets(train_y)
        # Load the validation data and normalize it
        val_x, val_y = get_mean_values_dataset(data_path + 'val/', shuffle=False)
        val_x, _ = minmax_normmalization(val_x, norm_x)
        val_y  = normalize_targets(val_y)
        # Get the values 
        train_x = train_x.values
        train_y = train_y.values
        val_x = val_x.values
        val_y = val_y.values


        # Generate model and define the optimizer
        model = Data_Model(loss = nn.MSELoss()).double()
        model.optimizer = optim.LBFGS(model.parameters(), history_size=300, max_iter=50)

    #elif(model_name == 'data_folder'):


    elif(model_name == 'image'):
        # Load the data
        train_x, train_y = load_images(data_path + 'train/')
        val_x, val_y = load_images(data_path + 'val/')    
        # Normalize the data and re arrange image dimensions
        train_x = np.transpose(train_x, (0, 3, 2, 1)) / 255 
        train_y = normalize_targets(pd.DataFrame(train_y))
        val_x = np.transpose(val_x, (0, 3, 2, 1)) / 255 
        val_y = normalize_targets(pd.DataFrame(val_y))
        # Get the numpy array containing the values
        train_y = train_y.values
        val_y = val_y.values

        # Create the model and define the optimizer
        model = Image_Model(loss = nn.L1Loss()).double()
        model.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)


    # In case we want to use the whole batch
    if(batch_size < 0):
        batch_size = len(train_x)

    # Create the folder
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")
    hiperparameters = model_name + '_e' + str(epochs) + '_bs' + str(batch_size) + '_lr' + str(lr) + '_' + extra
    save_dir = str(pathlib.Path().absolute()) + '/checkpoints/' + now + '_' + hiperparameters 
    os.mkdir(save_dir)

    # Add plot data
    model.save_dir = save_dir
    model.hiperparameters = hiperparameters
    
    # Verify the model and data
    print(model)
    print('Using ' + str(len(train_x)) + ' training samples')
    print('Using ' + str(len(val_x)) + ' validation samples')
    # Do the training
    model.start_training(save_dir, train_x, train_y, val_x, val_y, epochs, batch_size, lr)



"""
    Function designed to be a wraper around model inference

"""

def infer(weights_file, target_dir, extra):

    # Create model 
    model = None
    isDataModel = True

    # Try to load the weights on one model,if it fails
    # try the other model
    try:
        # Try image model
        model = Image_Model(loss = nn.L1Loss()).double()
        model.load_state_dict(torch.load(weights_file, map_location={'cuda:0': 'cpu'}))
        model.eval()
        # Mark that we are using image model
        isDataModel = False  
    except:
        # Try data model
        model = Data_Model(loss = nn.MSELoss()).double()
        model.load_state_dict(torch.load(weights_file))
        model.eval()    
    finally:
        if(model is None):
            sys.exit("Weights uncompatible with either model")
        else:
            print(model)



    if('pixelwise' in extra):
        # Load the training data and normalize it
        test_x, test_y, size  = get_pixelwise_mean_values(data_path + 'test/')
        test_x, norm_x = minmax_normmalization(test_x)
        test_y = normalize_targets(test_y)
        # Get the values 
        test_x = test_x.values
        test_y = test_y.values

        # Start testing
        model.pixelwise_test(test_x, test_y, data_path, size)


    elif(isDataModel):

        # Load the training data and normalize it
        test_x, test_y = get_mean_values_dataset(data_path + 'test/', shuffle=False)
        test_x, norm_x = minmax_normmalization(test_x)
        test_y = normalize_targets(test_y)
        # Get the values 
        test_x = test_x.values
        test_y = test_y.values

        # Start testing
        model.start_test(test_x, test_y, data_path)

    else:
        # Load the data
        test_x, test_y = load_images(data_path + 'test/')
        # Normalize the data and re arrange image dimensions
        test_x = np.transpose(test_x, (0, 3, 2, 1)) / 255 
        test_y = normalize_targets(pd.DataFrame(test_y))
        # Get the numpy array containing the values
        test_y = test_y.values

        # Start testing
        model.start_test(test_x, test_y, data_path)






    
if __name__ == "__main__":
    
    model = ''
    action = ''
    data_path = ''
    extra = ''

    try:
        # We want to recognize m, a, d as options with argument thats why 
        # the : follows them, h doesnt need arguments so its alone
        opts, args = getopt.getopt(sys.argv[1:],"hm:a:d:e:",["model=","action=","data=","extra="])
    except getopt.GetoptError:
        print('python3 train.py -m <model> -a <action> -d <data path> \nInference looks for the latest folder with the model name, training creates this folder and saves the weigths there')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            sys.exit('python3 train.py -m <model> -a <action> -d <data path> \nInference looks for the latest folder with the model name, training creates this folder and saves the weigths there')

        elif opt in ("-m", "--model"):
            if((arg == 'data') or (arg == 'data_folder') or (arg == 'image') or (os.path.isfile(arg))):
                model = arg
            else:
                sys.exit('Unavailable model, please choose data, image or a model weight')

        elif opt in ("-a", "--action"):
            if((arg == 'train') or (arg == 'infer')):
                action = arg
            else:
                sys.exit('Unavailable mode, please choose train or infer')
            
            
        elif opt in ("-d", "--data"):
            data_path = arg

        elif opt in ("-e", "--extra"):
            extra = arg


    print('Selected ' + model + ' model for ' + action + ', using the data stored in ' + data_path)
    

    if (action == 'train'):
        train(model, data_path, extra)
    else:
        infer(model, data_path, extra)
    

# Training

# Data model
# python3 train.py -m data -a train -d ../Images/definitive/o1_marked/ -e LBFGS_MSE_dataset1

# python3 train.py -m data -a train -d ../Images/definitive/o2_marked/ -e LBFGS_MSE_dataset2

# python3 train.py -m data -a train -d ../Images/definitive/o_marked/ -e LBFGS_MSE_Full_dataset

# Image model
# python3 train.py -m image -a train -d ../Images/definitive/o1_fused/ -e Adam_MAE_dataset1

# python3 train.py -m image -a train -d ../Images/definitive/o2_fused/ -e Adam_MAE_dataset2

# python3 train.py -m image -a train -d ../Images/definitive/o_fused/ -e Adam_MAE_Full_dataset


# Inference

# Data model
# python3 train.py -m checkpoints/03-06-2020_18:48:30_data_e70_bs999_lr0.0001_LBFGS_MSE_dataset1/data.pth -a infer -d ../Images/definitive/o1_marked/

# python3 train.py -m checkpoints/03-06-2020_19:22:36_data_e100_bs1008_lr0.0001_LBFGS_MSE_dataset2/data.pth -a infer -d ../Images/definitive/o2_marked/

# python3 train.py -m checkpoints/03-06-2020_20:19:12_data_e100_bs1979_lr0.0001_LBFGS_MSE_Full_dataset/data.pth -a infer -d ../Images/definitive/o_marked/

# python3 train.py -m checkpoints/03-06-2020_19:22:36_data_e100_bs1008_lr0.0001_LBFGS_MSE_dataset2/data.pth -a infer -d ../Images/definitive/ort_big_marked/ -e pixelwise

# Image model
# python3 train.py -m checkpoints/04-06-2020_03:57:57_image_e500_bs128_lr0.0001_Adam_MAE_dataset1/images.pth -a infer -d ../Images/definitive/o1_fused/

# python3 train.py -m checkpoints/22-05-2020_07:38:34image_e1000_bs64_lr0.0001/images.pth -a infer -d ../Images/definitive/o2_fused/

# python3 train.py -m checkpoints/22-05-2020_07:38:34image_e1000_bs64_lr0.0001/images.pth -a infer -d ../Images/definitive/o_fused/