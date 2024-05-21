import os
import cv2
import shutil
import numpy as np

def process(subdir,filename, image, output_directory):
    # Making directory 
    if not os.path.exists(os.path.join(output_directory, subdir)):
        os.makedirs(os.path.join(output_directory, subdir))
    output_subdir = os.path.join(output_directory, subdir)

    #Graysclaing, resizing, blurring
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (256, 256))
    cv2.imwrite(os.path.join(output_subdir, filename[:-4]+'_gray.png'), gray_image)

    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    cv2.imwrite(os.path.join(output_subdir, filename[:-4]+'_blurred.png'), blurred_image)
    
    #Noise
    noise = np.random.normal(0, 25, resized_image.shape).astype(np.uint8)
    noised_image = cv2.add(resized_image, noise)
    cv2.imwrite(os.path.join(output_subdir, filename[:-4]+'_noised.png'), noised_image)
    denoised_image = cv2.fastNlMeansDenoising(noised_image, None, 30, 7, 21)
    cv2.imwrite(os.path.join(output_subdir, filename[:-4]+'_noised-denoised.png'), denoised_image)

    #detecting edges
    edges = cv2.Canny(resized_image, 100, 200)
    cv2.imwrite(os.path.join(output_subdir, filename[:-4]+'_edges.png'), edges)

    #Histogram equalizing
    hist_equalized = cv2.equalizeHist(resized_image)
    cv2.imwrite(os.path.join(output_subdir, filename[:-4]+'_hist_equalized.png'), hist_equalized)

    #Thresholding
    _, global_thresh = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_subdir, filename[:-4]+'_global_thresholded.png'), global_thresh)
    
    adaptive_thresh = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(output_subdir, filename[:-4]+'_adaptive_thresholded.png'), adaptive_thresh)



def clear_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Iterate over all the files and directories inside the specified directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The directory {directory_path} does not exist.')
