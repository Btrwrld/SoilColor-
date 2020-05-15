from os import listdir
import numpy as np
import cv2, csv


def calc_rgb_mean(image):
    # If we read usig cv2 imread the value order is BGR
    return [np.mean(image[:, :, 0]), 
            np.mean(image[:, :, 1]),
            np.mean(image[:, :, 2])]



def start(path, save_dir):


    # Read all the base images available
    images = listdir(path)
    # Read all the marked images availabe
    target_images = listdir(save_dir)

    pending_removal = []
    # Remove the non image files
    for i in range( len(target_images) ):
        # If it isnt a png then remove it
        # If we dont want to use the references then remove them
        if(not('png' in target_images[i]) 
            or ('ref' in target_images[i])
            or ('csv' in target_images[i])):
            #print(target_images[i])
            pending_removal.append(i)
    # Remove the images
    for pr in pending_removal[::-1]:
        target_images.pop(pr)
    # Reset the variable for future use
    pending_removal = []

    # Iterate over each base image to search for the
    # targets associated targets
    c = 0
    lti = len(target_images)
    for image in target_images:
        # Remove the extension from the image name
        image = image.split('png')[0][:-1]
        
        #image = image[:-5]
        print(image)

        # Since we are using a naming convention
        # we know the reference name
        ref_B1_mean = calc_rgb_mean(cv2.imread(save_dir + image + '_refB1.png'))
        ref_D1_mean = calc_rgb_mean(cv2.imread(save_dir + image + '_refD1.png'))
        ref_E1_mean = calc_rgb_mean(cv2.imread(save_dir + image + '_refE1.png'))

        # Get ready to recive info about the chips
        chips = []
        R_mean = []
        G_mean = []
        B_mean = []

        # Go through every target
        '''for t in range(len(target_images)):

            # Get the target name
            target_name = target_images[t]

            # Check if the target is part of the 
            # current image, if so procees it
            if(image in target_name):

                # Read the image in BGR
                target = cv2.imread(save_dir + target_name)

                # Add the target name
                chips.append(target_name)
                # Get the components mean value
                mean_values = calc_rgb_mean(target)
                B_mean.append(mean_values[0])
                G_mean.append(mean_values[1])
                R_mean.append(mean_values[2])

                # Add the image to the removal list
                pending_removal.append(t)'''

        # Read the image in BGR
        target = cv2.imread(save_dir + image + '.png')

        # Add the target name
        chips.append(image + '.png')
        # Get the components mean value
        mean_values = calc_rgb_mean(target)
        B_mean.append(mean_values[0])
        G_mean.append(mean_values[1])
        R_mean.append(mean_values[2])

        # Remove the used targets
        for pr in pending_removal[::-1]:
            target_images.pop(pr)
        # Reset the variable for future use
        pending_removal = []

        l = len(chips)
        # Save the csv file for the image
        rows = zip( chips, 
                    R_mean, G_mean, B_mean, 
                    l*[ref_B1_mean[2]], l*[ref_B1_mean[1]], l*[ref_B1_mean[0]],
                    l*[ref_D1_mean[2]], l*[ref_D1_mean[1]], l*[ref_D1_mean[0]],
                    l*[ref_E1_mean[2]], l*[ref_E1_mean[1]], l*[ref_E1_mean[0]])

        with open(save_dir + image + '_MeanValues.csv', "w") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)

        c += 1
        print('Marked ' + str(c) + '/' + str(lti))
        





if __name__ == '__main__':

    # /home/erick/google_drive/PARMA/SoilColor/Images/outdoor 1/1_GLEY1_R_WBA_M.jpg
    print(10*'-' + 'Welcome to the soil color mean rgb generation tool' + 10*'-')
    # Ask for the images path
    path = '/home/erick/google_drive/PARMA/SoilColor/Images/ort_marked_small/'
    save_dir = '/home/erick/google_drive/PARMA/SoilColor/Images/ort_marked_small/'
    print('USING DEFUALT VALUES OF PATH AND SAVE')
    print('Path: ' + path)
    print('Save: ' + save_dir)
    # Start generator script
    start(path, save_dir)