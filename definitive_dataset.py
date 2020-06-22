import random, shutil, os

from tools.data_manager import *



def merge_csvs(source_dirs):

    # Build dataframe
    csvs = pd.DataFrame()

    # Hot fix for verification
    if(type(source_dirs) is str):
        source_dirs = [source_dirs]

    # Read all the csv files in each folder
    for sd in source_dirs:
        # Read all csv's
        files = os.listdir(sd)
        #print(str(files) + '\n')
        # Select only the csv's called mean vales
        files = select_valid_files(files, ['MeanValues'])
        # For each file
        for f in files:
            new_csv = pd.read_csv(sd + f, header=None)
            new_csv[len(new_csv.columns.values)] = sd.split('/')[-2] + '/'
            # Read it and append
            csvs = pd.concat([csvs, new_csv], ignore_index = True) 


    return csvs


def calc_partition_pos(max_len, train_perc, val_size, test_perc):

    training_lim = int(train_perc * max_len)
    val_lim = int((val_size * max_len) + training_lim)
    test_lim = int(test_perc * max_len + val_lim)
    
    return training_lim, val_lim, test_lim


'''
    If an element in the imgs list is in the source dir
    then copy it to the target dir and remove it from the imgs list
'''
def copy_if_in_source(source_dir, target_dir, imgs):

    # Get the elements in source
    source_imgs = os.listdir(source_dir)
    pending_removal = []

    # For each element in imgs
    for i in range(len(imgs)):
        # Flag to check if the img is present 
        isPresent = False
        # Get the name and remove the extension
        im_name = imgs[i].split('png')[0][:-1]
        # If it is in the source files
        if(any(im_name in s for s in source_imgs)):
            # Copy the file
            im_name += '_WRef.png'
            shutil.copy(source_dir + im_name , target_dir + im_name)
            # Update flag
            isPresent = True

        # Add to pending removal
        if(isPresent):
            pending_removal.append(i)

    # Remove the images
    for i in pending_removal[::-1]:
        imgs.pop(i)

    # Return the updated list
    return imgs

    
def generate_csv_dataset(source_dirs, target_dir, train_perc, val_size, test_perc):

    # Get all the data in the csvs
    csvs = merge_csvs(source_dirs)
    # Shuffle the rows 
    csvs = shuffle_df(csvs)    
    # Get the limits for each partition
    tr_l, v_l, _ = calc_partition_pos(len(csvs), train_perc, val_size, test_perc)
    # Create folders 
    os.mkdir(target_dir  )
    os.mkdir(target_dir + 'train/')
    os.mkdir(target_dir + 'val/')
    os.mkdir(target_dir + 'test/')
    # Write partitions
    csvs.iloc[:tr_l].to_csv(    target_dir + 'train/train_MeanValues.csv', 
                                header=False, index=False)
    csvs.iloc[tr_l: v_l].to_csv(target_dir + 'val/val_MeanValues.csv', 
                                header=False, index=False)
    csvs.iloc[v_l:].to_csv(     target_dir + 'test/test_MeanValues.csv', 
                                header=False, index=False)

    # Return the list of all the chip images
    # As        training                    validation              testing
    return csvs.iloc[:tr_l, 0].values, csvs.iloc[tr_l:v_l, 0].values, csvs.iloc[v_l:, 0].values




def generate_image_dataset(source_dirs, target_dir, imgs):

    # Generate the target dirs
    target_dirs = [ target_dir + 'train/',
                    target_dir + 'val/',
                    target_dir + 'test/']
    # Create directories
    os.mkdir(target_dir)
    os.mkdir(target_dirs[0])
    os.mkdir(target_dirs[1])
    os.mkdir(target_dirs[2])

    # For each list in imgs(train list, val list and test list)
    for i in range(3):
        # Check each source dir
        for sd in source_dirs:
            # Create directories
            td = target_dirs[i] + sd.split('/')[-2] + '/'
            os.mkdir(td)
            # Copy if present and update
            imgs[i] = copy_if_in_source(sd, td, imgs[i])


# Checks that every image in each folder has its counterpart in the csv
def verify_dataset(image_target, data_target):

    image_targets = [image_target + 'train/',
                    image_target + 'val/',
                    image_target + 'test/']
    data_targets = [data_target + 'train/',
                    data_target + 'val/',
                    data_target + 'test/']

    # Check all data is in the imgs
    for i in range(3):
        # Read the csv and get the img names
        df = merge_csvs(data_targets[i])
        data = df.iloc[:,0].values
        # Image container
        images = []
        print(df.iloc[:,-1].unique())
        for src in df.iloc[:,-2].unique():
            # Follow the naming convention
            src = src.split('_')[0] + "_fused"
            # Read the images
            images += os.listdir(image_targets[i] + src)
        for img_data in data:
            # Remove the extension
            img_data = img_data.split('png')[0][:-1]
            # If it is not present in the files, then so something
            if(not(any(img_data in s for s in images))):
                print('La imagen de ' + str(i) + ' ' + img_data + ' no está en la carpeta de imagenes')

    # Check all imgs is in the data
    for i in range(3):
        # Read the csv and get the img names
        df = merge_csvs(data_targets[i])
        data = df.iloc[:,0].values
        # Image container
        images = []
        for src in df.iloc[:,-2].unique():
            # Follow the naming convention
            src = src.split('_')[0] + "_fused"
            # Read the images
            images += os.listdir(image_targets[i] + src)
        for img_images in images:
            # Remove the extension and the _WRef
            img_images = img_images[:-9]
            # If it is not present in the files, then so something
            if(not(any(img_images in s for s in data))):
                print('Los datos de ' + str(i) + ' ' + img_images + ' no está en la carpeta de datos')

    
def generate_definitive_dataset(csv_source, images_source, csv_target, images_target,
                                train_perc, val_size, test_perc):

    # First generate the csv dataset
    print('Generating csv dataset..')
    train, val, test = generate_csv_dataset(csv_source, csv_target, train_perc, val_size, test_perc)
    # Then the image dataset
    print('Generating image dataset')
    generate_image_dataset(images_source, images_target, [list(train), list(val), list(test)])
    # Verify everything is ok
    print('Verifying dataset integrity')
    verify_dataset(images_target, csv_target)


if __name__ == "__main__":

    images_source = ['/home/erick/google_drive/PARMA/SoilColor/Images/o1_fused/',
                    '/home/erick/google_drive/PARMA/SoilColor/Images/o2_fused/']

    csv_source = [  '/home/erick/google_drive/PARMA/SoilColor/Images/o1_marked/',
                    '/home/erick/google_drive/PARMA/SoilColor/Images/o2_marked/']
    csv_target = '/home/erick/google_drive/PARMA/SoilColor/Images/definitive/o_marked/'
    images_target = '/home/erick/google_drive/PARMA/SoilColor/Images/definitive/o_fused/'

    
    #generate_csv_dataset(csv_source, csv_target, 0, 0, 1)


    generate_definitive_dataset(csv_source, images_source, csv_target, images_target, 0.7, 0.2, 0.1)

