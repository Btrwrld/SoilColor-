import numpy as np 
from tools.data_manager import *

# Read the image 
ort = np.genfromtxt('../Images/outdoor_realcase.csv',delimiter=',',skip_header=1)
# Remove the item numer, archeologist  and image number columns
ort = ort[:, 1:-2]
# Permute columns to order as HCV
ort[:, [1, 2]] = ort[:, [2, 1]]



# Add all the values
ort_res = ort[:16, :] + ort[16:32, :] + ort[32:48, :]
# Calc the mean value
ort_res /= 3

# Write the values in order HCV
write_csv('../Images/outdoor_realcase_mean.csv', zip(ort_res[:,0], ort_res[:,1], ort_res[:,2]))
