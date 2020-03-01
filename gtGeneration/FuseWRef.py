from os import listdir
import numpy as np
import cv2, csv

import matplotlib.pyplot as plt

def start(path, pathsave_dir):


    # Read all the base images available
    images = listdir(path)
    # Read all the marked images availabe
    target_images = listdir(save_dir)

    pending_removal = []
    # Remove the non image files
    for i in range( len(images) ):
        # If it isnt a png then remove it
        # If we dont want to use the references then remove them
        if(not('png' in images[i]) or ('ref' in images[i])):
            pending_removal.append(i)
    
    # Remove the images
    for pr in pending_removal[::-1]:
        images.pop(pr)
    # Reset the variable for future use
    pending_removal = []

    # Iterate over each base image to search for the
    # targets associated targets
    c = 0
    for image in images:
        # Remove the extension from the image name
        # and remove the color va
        image = image.split('png')[0][:-1]
        core_name = image.split('v')[0]

        # Since we are using a naming convention
        # we know the reference name
        ref_B1 = cv2.imread(path + core_name + 'refB1.png')
        ref_D1 = cv2.imread(path + core_name + 'refD1.png')
        ref_E1 = cv2.imread(path + core_name + 'refE1.png')

        # Crop the images
        heigth = ref_B1.shape[1] // 3
        center = ref_B1.shape[1] // 2
        ref_B1 = ref_B1[center - heigth : center + heigth, :, :]
        ref_D1 = ref_D1[center - heigth : center + heigth, :, :]
        ref_E1 = ref_E1[center - heigth : center + heigth, :, :]

        # Read the image in BGR
        fusion = cv2.imread(path + image + '.png')
        # Fuse the images
        fusion = np.vstack( (fusion, ref_B1, ref_D1, ref_E1) )
        # Store the fusion
        cv2.imwrite(save_dir + image + '_WRef' + '.png', fusion)


        c += 1
        print('Fused ' + str(c) + '/' + str(len(images)))
        





if __name__ == '__main__':

    # /home/erick/google_drive/PARMA/SoilColor/Images/outdoor 1/1_GLEY1_R_WBA_M.jpg
    print(10*'-' + 'Welcome to the soil color mean rgb generation tool' + 10*'-')
    # Ask for the images path
    path = '/home/erick/google_drive/PARMA/SoilColor/Images/o2_marked/'
    save_dir = '/home/erick/google_drive/PARMA/SoilColor/Images/o2_fused/'
    print('USING DEFUALT VALUES OF PATH AND SAVE')
    print('Path: ' + path)
    print('Save: ' + save_dir)
    # Start generator script
    start(path, save_dir)