import subprocess
import glob
import os
import cv2
import numpy as np
import natsort

def crop_images(dataset_folder_path='./dataset/val/',
                fe_path='/home/eelassanai/Workspace/openface/OpenFace/build/bin/FeatureExtraction'):
    aligned_path = dataset_folder_path + 'aligned/'
    if not os.path.exists(aligned_path):
        os.mkdir(aligned_path)

    for subjectName in glob.glob(dataset_folder_path + 'images/*'):
        dataset_rawpic = subjectName

        # Create new directory for each subject
        dir_crop_sub = aligned_path + str(subjectName.split('/')[-1]) + '/'
        if not os.path.exists(dir_crop_sub):
            #shutil.rmtree(dir_crop_sub)
            os.mkdir(dir_crop_sub)
        print('Subject', subjectName.split('/')[-1])
        # 256*256, npy, h5,
        subprocess.run([fe_path, "-fdir", dataset_rawpic, "-out_dir", dir_crop_sub, "-nomask"])

def face_alignment(fe_path, dataset_rawpic, dir_crop_sub, faceSize=224):
    subprocess.run([fe_path, "-fdir", dataset_rawpic, "-out_dir", dir_crop_sub, "-simsize", str(faceSize), "-nomask"])
    subprocess.run(["rm", "-rf", dataset_rawpic])

def convert_directory_to_image_file(folder_path='dataset/val/aligned/19874/', filename='19874'):
    aligned_dir = folder_path + filename + '_aligned/'
    images=[]
    for dir_sub_vid_img in natsort.natsorted(glob.glob(aligned_dir + "frame*.bmp")):
        images.append(cv2.imread(dir_sub_vid_img, 1))
    images = np.stack(images)
    subprocess.run(["rm", "-rf", folder_path])
    return images

def delete_folder(folder_path):
    subprocess.run(["rm", "-rf", folder_path])

if __name__ == '__main__':
    #fe_path: path to installed openFace feature extraction
    #crop_images(dataset_folder_path='./dataset/val/', fe_path='')
    convert_directory_to_image_file('dataset/aligned/02127/', '02127')
