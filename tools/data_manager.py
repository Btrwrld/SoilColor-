import csv, random

import numpy as np
import pandas as pd 
from os import listdir
from sklearn import preprocessing
from skimage.io import imread
from skimage.transform import resize

# Definition of normalizated hue, value and chroma values used for later classification
hue_norm = np.array([0, 0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333, 1])
chroma_norm = np.array([0, 0.14285714, 0.28571429, 0.42857143, 0.71428571, 1])
value_norm = np.array([0, 0.09090909, 0.27272727, 0.45454545, 0.63636364,0.81818182, 1])

hue_val = np.array([10, 12.5, 15, 17.5, 20, 22.5, 25])
val_val = np.array([8, 7, 6, 5, 4, 3, 2.5])
chr_val = np.array([8, 6, 4, 3, 2, 1, 1])
m = np.array([hue_val, chr_val, val_val])
target_normalizer = preprocessing.MinMaxScaler()
target_normalizer.fit(m.T)

def _map_hue(hue):
    # Convert from values like 10R to 10.0 or similar
    hue_table = {   '10R'   :   10.0,
                    '2.5YR' :   12.5,
                    '5YR'   :   15.0,
                    '7.5YR' :   17.5,
                    '10YR'  :   20.0,
                    '2.5Y'  :   22.5,
                    '5Y'    :   25.0
                }

    # If the hue value is found return its definition, else return NaN
    return hue_table.get(hue, float('NaN'))

def load_mean_values(path, file_name):

    # We know the mean values files end with that name 
    data = pd.read_csv(path + file_name, header=None) 

    # Now we divide the dataset and extract the targets 
    # from the rest
    targets = data.iloc[:, 0]
    data = data.iloc[:, 1:]

    # Process the targets to separate into the Hue-Chroma-Value
    # using the names of the rows
    targets = targets.values.tolist()
    data_to_remove =[]
    # Since one card is a constant hue, we just extract it, 
    # cast it to float and replicate it many times
    hue = []
    value = []
    chroma = []
    for i in range(len(targets)):
        # If we read an invalid hue then jump the line and drop the data column
        poss_hue = _map_hue(targets[i].split('_')[1])
        if(np.isnan(poss_hue)):
            data_to_remove.append(i)
            continue
        hue.append(poss_hue)
        # Get the components
        components = targets[i].split('_')[5][:-4].split('c')
        # Extract the values we need
        chroma.append(float(components[1]))
        value.append(float(components[0][1:]))

    # Remove the invalid images
    for i in data_to_remove[::-1]:
        data = data.drop(data.index[i])

    # Create a dataframe with the target values
    targets = pd.DataFrame.from_dict( { 'Hue'   :   hue,
                                        'Chroma' :  chroma,
                                        'Value' :   value})
    # Set column names to data
    data.columns = ['R_mean', 'G_mean', 'B_mean', 
                    'ref_B1_R_mean', 'ref_B1_G_mean', 'ref_B1_B_mean', 
                    'ref_D1_R_mean', 'ref_D1_G_mean', 'ref_D1_B_mean', 
                    'ref_E1_R_mean', 'ref_E1_G_mean', 'ref_E1_B_mean']


    return data, targets


# Takes an order and shuffles a df according to it
def shuffle_df(df, new_order=[]):

    if(len(new_order) == 0):
        new_order = np.arange(df.shape[0])
        np.random.shuffle(new_order)
    
    df = df.reindex(new_order)
    return df.reset_index(drop=True)

def get_dataframe_skeleton():

    # Start with empty data frames
    targets = pd.DataFrame.from_dict( { 'Hue'   :   [],
                                        'Chroma' :   [],
                                        'Value' :   []})

    data = pd.DataFrame.from_dict({ 'R_mean'        :   [],
                                    'G_mean'        :   [],
                                    'B_mean'        :   [],
                                    'ref_B1_R_mean' :   [],
                                    'ref_B1_G_mean' :   [],
                                    'ref_B1_B_mean' :   [],
                                    'ref_D1_R_mean' :   [],
                                    'ref_D1_G_mean' :   [],
                                    'ref_D1_B_mean' :   [],
                                    'ref_E1_R_mean' :   [],
                                    'ref_E1_G_mean' :   [],
                                    'ref_E1_B_mean' :   []})

    return data, targets


# Return the shuffled data and the targets
def get_mean_values_dataset(path, shuffle=True):

    # Get the csv files that contain MeanValues in their name
    mean_values = listdir(path)
    # Select only the csv's called mean vales
    mean_values = select_valid_files(mean_values, ['MeanValues'], 0)

    # Start with empty data frames
    data, targets = get_dataframe_skeleton()

    # Start the data collection
    for image in mean_values:
        # Get the values
        d, t = load_mean_values(path, image)
        # If d and t are empty lists then it means that we are reading 
        # a card that isn't valid so dont add anything and continue
        if((len(d) == 0) and (len(t) == 0)):
            continue
        # Append the data
        targets = pd.concat([targets, t], ignore_index = True) 
        data = pd.concat([data, d], ignore_index = True) 

    # If shuffle flag is active, shuffle the df
    if(shuffle):
        # Generate a new order
        new_order = np.arange(targets.shape[0])
        np.random.shuffle(new_order)
        # Shuffle the values
        targets = shuffle_df(targets, new_order)
        data = shuffle_df(data, new_order)

    
    # Return all 
    return data, targets


def normalize_targets(df):
    return pd.DataFrame(target_normalizer.transform(df.values), columns=df.columns)


# Recieves a pandas df and normalizes column wise
def minmax_normmalization(df, minmax_scaler=None):

    # Add already calculated minmax scaler support
    if(minmax_scaler == None):
        # Create the scaler
        minmax_scaler = preprocessing.MinMaxScaler()
        # Train it
        minmax_scaler.fit(df.values)

    # Transform the data
    df = pd.DataFrame(minmax_scaler.transform(df.values), columns=df.columns)

    # Return the normalized data and the normalizator to compute the inverse
    return df, minmax_scaler










############################## Image related stuff ##############################

'''
    Takes a dirty list and a list with a number of valid formats, then it
    iterates over the dirty list checking if the files of the dirty lists
    belong to any of the valid formats, and if they don't, they get removed

    dirty_list:     List with the values we'll check
    valid_ids:      List with the valid identifiers
    checK_ext:      If 1 checks the extension, if 0 checks the name


'''
def select_valid_files(dirty_list, valid_ids=['png', 'jpg', 'jpeg'], check_ext=1):

    # Here we'll store the indexes of the 
    # elements we want to remove
    pending_removal = []

    # For each element in the list
    for i in range(len(dirty_list)):

        # Get the element
        e = dirty_list[i]
        # Create the flag
        isValid = False

        # If its a file then check if it has valid format
        if('.' in e):
            # Check each valid format so if the format is present
            # in the element, then its valid
            for vf in valid_ids:
                # Check if present
                if(vf in e):
                    # If it is then update the flag
                    isValid = True
                    # Stop the loop
                    break

        # If it is a folder, just remove it

        # if the value is not valid, then add it's index 
        # to pending removal
        if(not(isValid)):
            pending_removal.append(i)


    # Clean the list
    for idx in pending_removal[::-1]:
        dirty_list.pop(idx)

    # Return the value
    return dirty_list


def select_valid_images(images):

    valid_images = []

    # Remove non image files
    images = select_valid_files(images)

    for img in images:

        hue = _map_hue(img.split('_')[1])

        # Check if we have a valid image (one wich hue value is 
        # defined in the map hue function)
        if(not(pd.isnull(hue))):
            valid_images.append(img)

    return valid_images

#   Image loader function, loads the images in the 
#   order they are given, can load batches or the whole list.
#   If no image names are given, the whole folder is explored and 
#   the asumption that the files inside are images is made
#
#   path:           Directory where we'll look for the images.
#   image_names:    List with the name of all the images. Default is []
#   batch_size:     Number of images to load, by default uses -1
#                   to load all the given images.
#
def load_images(path, image_names=[], batch_size=-1):

    # If we have no image names the images on the path
    if(len(image_names) == 0):
        # Read all the images on the path 
        image_names = listdir(path)
        # Select the valid images
        image_names = select_valid_images(image_names)
        # Shuffle 'em
        random.shuffle(image_names)


    x = []
    y = []

    if(batch_size == -1):
        batch_size = len(image_names)

    for _ in range(batch_size):

        # Read the image and resize it to 40x40
        name = image_names.pop(0)
        image = resize(imread(path + name), (40, 40), anti_aliasing=True)
        
        # Get the color information
        hue = _map_hue(name.split('_')[1])
        c = name.split('_')[5]
        v = float(c.split('c')[0][1:])
        c = float(c.split('c')[1])

        # Add it to the image list 
        x.append(image)
        y.append([hue, c, v])


    # Return x and y as a numpy array
    x = np.array(x)
    y = np.array(y)

    return x, y



'''
    Takes a direction to create a csv file in and a zip of values
    as rows
 
'''
def write_csv(save_dir, rows):
    with open(save_dir, "w") as f:
                w = csv.writer(f)
                for row in rows:
                    w.writerow(row)
    
    