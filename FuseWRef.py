from os import listdir
import numpy as np
import cv2, csv


import matplotlib.pyplot as plt
from skimage.transform import resize
from tools.data_manager import select_valid_files

def start(path, save_dir):

    img_extension = 'jpg'


    # Read all the marked images availabe
    target_images = listdir(path)

    target_images = select_valid_files(target_images, valid_ids=[img_extension])

    # Iterate over each base image to search for the
    # targets associated targets
    c = 0
    for image in target_images:
        # Remove the extension from the image name
        # and remove the color va
        image = image.split(img_extension)[0][:-1]
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
        fusion = resize(cv2.imread(path + image + '.' +img_extension), 
                        (ref_B1.shape[0], ref_B1.shape[1]), 
                        anti_aliasing=True)
        # Fuse the images
        fusion = np.vstack( (fusion, ref_B1, ref_D1, ref_E1) )
        # Store the fusion
        cv2.imwrite(save_dir + image + '_WRef' + '.png', fusion)


        c += 1
        print('Fused ' + str(c) + '/' + str(len(target_images)))
        





if __name__ == '__main__':

    # /home/erick/google_drive/PARMA/SoilColor/Images/outdoor 1/1_GLEY1_R_WBA_M.jpg
    print(10*'-' + 'Welcome to the soil color mean rgb generation tool' + 10*'-')
    # Ask for the images path
    path = '/home/erick/google_drive/PARMA/SoilColor/Images/ort_marked_big/'
    save_dir = '/home/erick/google_drive/PARMA/SoilColor/Images/definitive/ort_big_fused/test/'
    print('USING DEFUALT VALUES OF PATH AND SAVE')
    print('Path: ' + path)
    print('Save: ' + save_dir)
    # Start generator script
    start(path, save_dir)