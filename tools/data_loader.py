import numpy as np
import pandas as pd 
from os import listdir
from sklearn import preprocessing
from skimage.io import imread
from skimage.transform import resize

# Definition of normalizated hue, value and chroma values used for later classification
hue_norm = np.array([0, 0.16666667, 0.33333333, 0.46666667, 0.5, 0.66666667, 0.83333333, 1])
chroma_norm = np.array([0, 0.14285714, 0.28571429, 0.42857143, 0.71428571, 1])
value_norm = np.array([0, 0.09090909, 0.27272727, 0.45454545, 0.63636364,0.81818182, 1])


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

def _load_mean_values(path, file_name):

    # We know the mean values files end with that name 
    data = pd.read_csv(path + file_name, header=None) 

    # Now we divide the dataset and extract the targets 
    # from the rest
    targets = data.iloc[:, 0]
    data = data.iloc[:, 1:]

    # Process the targets to separate into the Hue-Croma-Value
    # using the names of the rows
    targets = targets.values.tolist()
    # Since one card is a constant hue, we just extract it, 
    # cast it to float and replicate it many times
    hue = [_map_hue(targets[0].split('_')[1])] * len(targets)
    value = []
    croma = []
    for img in targets:
        # Get the components
        components = img.split('_')[5][:-4].split('c')
        # Extract the values we need
        croma.append(float(components[1]))
        value.append(float(components[0][1:]))


    # Create a dataframe with the target values
    targets = pd.DataFrame.from_dict( { 'Hue'   :   hue,
                                        'Croma' :   croma,
                                        'Value' :   value})
    # Set column names to data
    data.columns = ['R_mean', 'G_mean', 'B_mean', 
                    'ref_B1_R_mean', 'ref_B1_G_mean', 'ref_B1_B_mean', 
                    'ref_D1_R_mean', 'ref_D1_G_mean', 'ref_D1_B_mean', 
                    'ref_E1_R_mean', 'ref_E1_G_mean', 'ref_E1_B_mean']


    return data, targets

# Return the shuffled data and the targets
def get_mean_values_dataset(path):

    # Get the csv files that contain MeanValues in their name
    mean_values = listdir(path)

    pending_removal = []
    # Remove the non image files
    for i in range( len(mean_values) ):
        # If it isnt a png then remove it
        # If we dont want to use the references then remove them
        if(not('MeanValues' in mean_values[i])):
            pending_removal.append(i)

    # Remove the images
    for pr in pending_removal[::-1]:
        mean_values.pop(pr)

    # Start with empty data frames
    targets = pd.DataFrame.from_dict( { 'Hue'   :   [],
                                        'Croma' :   [],
                       get_mean_values_dataset                 'Value' :   []})

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
    # Start the data collection
    for image in mean_values:
        # Get the values
        d, t = _load_mean_values(path, image)
        # Append the data
        targets = targets.append(t, ignore_index = True, sort=False) 
        data = data.append(d, ignore_index = True, sort=False) 


    # Shuffle the values
    new_order = np.arange(targets.shape[0])
    np.random.shuffle(new_order)
    targets = targets.reindex(new_order)
    targets = targets.reset_index(drop=True)
    data = data.reindex(new_order)
    data = data.reset_index(drop=True)

    
    # Return all 
    return data, targets

# Recieves a pandas df and normalizes column wise
def minmax_normmalization(df):
    # Create the scaler
    minmax_scaler = preprocessing.MinMaxScaler()
    # Fit the model and transform the data
    df = pd.DataFrame(minmax_scaler.fit_transform(df.values), columns=df.columns)

    # Return the normalized data and the normalizator to compute the inverse
    return df, minmax_scaler








def select_valid_images(images):

    valid_images = []

    for img in images:

        hue = _map_hue(img.split('_')[1])

        # Check if we have a valid image (one wich hue value is 
        # defined in the map hue function)
        if(not(pd.isnull(hue))):
            valid_images.append(img)

    return valid_images

#   Image loader function, loads the images in the 
#   order they are given, can load batches or the whole list.
#
#   path:           Directory where we'll look for the images.
#   image_names:    List with the name of all the images.
#   batch_size:     Number of images to load, by default uses -1
#                   to load all the given images.
#
def load_images(path, image_names, batch_size=-1):

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

    
    