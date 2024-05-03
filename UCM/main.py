import os
import numpy as np
import cv2
import natsort
from color_equalisation import RGB_equalisation
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')

if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/NonPhysical/UCM"
folder = "C:/Users/Dell/Documents/Thesis-CrackDetection/"
input_directory = os.path.join(folder, "InputImages")
output_directory = "./OutputImages"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

files = os.listdir(input_directory)
files = natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = os.path.join(input_directory, file)
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********', file)
        img = cv2.imread(filepath)
        sceneRadiance = RGB_equalisation(img)
        sceneRadiance = stretching(sceneRadiance)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)
        cv2.imwrite(os.path.join(output_directory, f'{prefix}_UCM.jpg'), sceneRadiance)
