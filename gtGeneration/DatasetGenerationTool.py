from os import listdir

import cv2, csv
import numpy as np
import matplotlib.pyplot as plt


# Define the length in pixels from the center to the border of the square
color_length = 4

# Create the data structures used to store the points
ref_center = {  'B1' : (0,0),
                'D1' : (0,0),
                'E1' : (0,0),
                'isFinished' : False}
next_ref = 'B1'

# The color is stored in a value#chroma# way
color_center = {}
# Store a list of the colors to be selected in a row-sorted way
pending_colors = []

# Store the current card name
card_name = ''

# Card values, used for iteration purposes only
values = ['8', '7', '6', '5', '4', '3', '2.5']
chromas = ['1', '2', '3', '4 ', '6', '8']


# Defines the click behavior on the image 
def onclick(event):
    if event.xdata != None and event.ydata != None:

        # Fill the reference center if needed
        if(not(ref_center['isFinished'])):

            # Declare we will be using global variables
            global next_ref
            # Update the dictionary
            ref_center[next_ref] = (int(event.xdata), int(event.ydata))
            
            # Update the next reference
            if(next_ref == 'B1'):
                next_ref = 'D1'
            elif(next_ref == 'D1'):
                next_ref = 'E1'
            else:
                ref_center['isFinished'] = True
                next_ref = 'B1'

        # Fill the colors
        else:

            # If all colors have been saved the close the image
            if(len(pending_colors) == 0):
                plt.close()

            else:
                # Select the color and remove it from the pending colors list
                color = pending_colors.pop(0)
                # Add it to the collected colors dictionary
                color_center[color] = (int(event.xdata), int(event.ydata))

        print((int(event.xdata), int(event.ydata)))

# Draws the image
def show_image(image, name):

    # Set up the canvas to dispay and capture clicks
    ax = plt.gca()
    fig = plt.gcf()
    fig.canvas.set_window_title(name)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Display the image to preview some values
    plt.show(block=False)

    # Select the color space
    select_color_space()

    # Link the click event listener to the function to get the coordinates
    fig.canvas.mpl_connect('button_press_event', onclick)

    # Display the image to capture values
    plt.show()


def select_color_space():

    # Capture the available color names
    for value in values:
        
        # Ask for the last chroma value
        c = input('On row ' + value + ', enter the last valid chroma value: ')
        
        # Check if the value is valid
        if(c in chromas):
            # Combine the values to generate the pending colors list
            for chroma in chromas[: chromas.index(c) + 1]:
                pending_colors.append('v' + value + 'c' + chroma)
        else:
            print('Invalid value')

    print('You can now begin the color capture process!')


def create_dataset(image, name, save_dir):

    print(ref_center)

    # Get the reference points 
    refB1 = image[  ref_center['B1'][1] - color_length : ref_center['B1'][1] + color_length,
                    ref_center['B1'][0] - color_length : ref_center['B1'][0] + color_length, :]
    refD1 = image[  ref_center['D1'][1] - color_length : ref_center['D1'][1] + color_length,
                    ref_center['D1'][0] - color_length : ref_center['D1'][0] + color_length, :]         
    refE1 = image[  ref_center['E1'][1] - color_length : ref_center['E1'][1] + color_length,
                    ref_center['E1'][0] - color_length : ref_center['E1'][0] + color_length, :]          

    # Save the references 
    cv2.imwrite(save_dir + name + '_refB1.png', refB1)
    cv2.imwrite(save_dir + name + '_refD1.png', refD1)
    cv2.imwrite(save_dir + name + '_refE1.png', refE1)

    # Reset the reference state to prepare next image
    ref_center['isFinished'] = False

    # Prepare the position data for the csv
    x = [ref_center['B1'][0], ref_center['D1'][0], ref_center['E1'][0]]
    y = [ref_center['B1'][1], ref_center['D1'][1], ref_center['E1'][1]]
    colors = ['B1', 'D1', 'E1'] + list(color_center.keys())
    print(colors)
    print(len(colors))

    # For each marked color
    for color in color_center.keys():

        # Extract the color
        color_image = image[color_center[color][1] - color_length : color_center[color][1] + color_length,
                            color_center[color][0] - color_length : color_center[color][0] + color_length, 
                            :]
        
        # Store the color position to the csv list
        x.append(color_center[color][0])
        y.append(color_center[color][1])

        # Store the color
        cv2.imwrite(save_dir + name + '_' + color + '.png', color_image)

    # Save the csv file for the image
    rows = zip(x, y, colors)
    with open(save_dir + name + '.csv', "w") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)




def start(path, save_dir):

    # Read all the images
    images = listdir(path)
    global card_name

    # Mark each image
    for im_name in images:
        # Read the image 
        image = cv2.imread(path + im_name)
        # Remove the extension
        im_name = im_name.split('.')[0]
        # Set the card name, thanks to the image naming convention
        # we know the card name always follow the first _
        card_name = im_name.split('_')[1]
        # Make a blocking show that closes the image when we are done marking
        show_image(image, im_name)
        # Create the gt for the image and reset store data structures
        create_dataset(image, im_name, save_dir)







if __name__ == '__main__':

    # /home/erick/google_drive/PARMA/SoilColor/Images/outdoor 1/1_GLEY1_R_WBA_M.jpg
    print(10*'-' + 'Welcome to the soil color gt generetion tool' + 10*'-')
    # Ask for the images path
    path = input('Insert image folder path: ')
    save_dir = input('Insert image save path: ')
    # Start generator script
    start(path, save_dir)