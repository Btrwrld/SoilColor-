from os import listdir
import numpy as np
import cv2, csv

def start(path, save_dir):


    # Read all the base images available
    images = listdir(path)
    # Read all the marked images availabe
    target_images = listdir(save_dir)

    pending_removal = []
    # Remove the non image files
    for i in range( len(target_images) ):
        # If it isnt a png then remove it
        if(not('png' in target_images[i]) ):
            pending_removal.append(i)
    # Remove the images
    for pr in pending_removal[::-1]:
        target_images.pop(pr)
    # Reset the variable for future use
    pending_removal = []

    # Iterate over each base image to search for the
    # targets assosiated targets
    c = 0
    for image in images:
        # Remove the extension from the image name
        image = image.split('jpg')[0][:-1]

        # Get ready to recive info about the chips
        chips = []
        R_mean = []
        G_mean = []
        B_mean = []

        # Go through every target
        for t in range(len(target_images)):

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
                B_mean.append(np.mean(target[:, :, 0]))
                G_mean.append(np.mean(target[:, :, 1]))
                R_mean.append(np.mean(target[:, :, 2]))

                # Add the image to the removal list
                pending_removal.append(t)

        # Remove the used targets
        for pr in pending_removal[::-1]:
            target_images.pop(pr)
        # Reset the variable for future use
        pending_removal = []

        # Save the csv file for the image
        rows = zip(chips, R_mean, G_mean, B_mean)
        with open(save_dir + image + '_MeanValues.csv', "w") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)

        c += 1
        print('Marked ' + str(c) + '/' + str(len(images)))
        





if __name__ == '__main__':

    # /home/erick/google_drive/PARMA/SoilColor/Images/outdoor 1/1_GLEY1_R_WBA_M.jpg
    print(10*'-' + 'Welcome to the soil color mean rgb generation tool' + 10*'-')
    # Ask for the images path
    path = '/home/erick/google_drive/PARMA/SoilColor/Images/o1_base/'
    save_dir = '/home/erick/google_drive/PARMA/SoilColor/Images/o1_marked/'
    print('USING DEFUALT VALUES OF PATH AND SAVE')
    print('Path: ' + path)
    print('Save: ' + save_dir)
    # Start generator script
    start(path, save_dir)