import os
import cv2
from helper import clear_directory, process

main_directory = 'pokemonDB'
output_directory = 'cv_output'

clear_directory(output_directory)


for subdir in os.listdir(main_directory):
    subdir_path = os.path.join(main_directory, subdir)
    
    for filename in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, filename)

        image = cv2.imread(file_path)
        cv2.destroyAllWindows()
        process(subdir, filename, image, output_directory)
        cv2.imshow(f'{subdir} - {filename}', image)
        cv2.waitKey(0)

cv2.destroyAllWindows()