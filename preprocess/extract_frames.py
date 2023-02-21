import cv2
import os
from extract_utils import face_alignment, convert_directory_to_image_file, delete_folder
from tqdm import tqdm
import numpy as np
from pathlib import Path

def extract_frames(dataset_folder_path='dataset/train/',
                   fe_path='/home/yfangba/Workspace/openface/OpenFace/build/bin/FeatureExtraction'):
    filenamePadding = 5
    aligned_path = dataset_folder_path + 'aligned/'
    if not os.path.exists(aligned_path):
        os.mkdir(aligned_path)
    directory = os.fsencode(dataset_folder_path+'mp4/')
    if not os.path.exists(dataset_folder_path + 'images'):
        os.makedirs(dataset_folder_path + 'images')
    final_saving_dir = dataset_folder_path + '/processed/'
    if not os.path.exists(final_saving_dir):
        os.mkdir(final_saving_dir)

    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)

        path = Path(final_saving_dir+filename[:-4]+'.npy')
        if path.is_file():
            print('Skipping ' + str(file))
            continue
        if not os.path.exists(dataset_folder_path + 'images/'+filename[:-4]):
            os.makedirs(dataset_folder_path + 'images/'+filename[:-4])
        if filename.endswith(".mp4"):
            vid = cv2.VideoCapture(dataset_folder_path + 'mp4/' + filename)
            index = 0
            while (True):
                ret, frame = vid.read()
                if not ret:
                    break
                indexStr = '0' * (filenamePadding - len(str(index))) + str(index)
                name = dataset_folder_path + '/images/'+filename[:-4]+'/frame' + indexStr + '.jpg'
                #print('Creating...' + name)
                cv2.imwrite(name, frame)
                index += 1

            # after we generate the directory containing images, we extract faces and delete this directory to save memory
            saving_dir = dataset_folder_path + '/images/'+filename[:-4]
            dest_dir = dataset_folder_path + '/aligned/'+filename[:-4]
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)
            #video_dir = dataset_folder_path+'mp4/'+filename
            face_alignment(fe_path, saving_dir, dest_dir)
            # read and save aligned faces, delete aligned directory
            #images = convert_directory_to_image_file(dest_dir+'/', filename[:-4])
            #np.save(final_saving_dir+filename[:-4], images)
        else:
            continue
    # delete align path and image path
    #delete_folder(aligned_path)
    #delete_folder(dataset_folder_path + 'images')

if __name__ == "__main__":
    extract_frames(dataset_folder_path='./dataset/train/')
    extract_frames(dataset_folder_path='./dataset/val/')
    # /home/yini/OpenFace/build/bin/FeatureExtraction
    # /data/abaw5/val/
