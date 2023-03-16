import cv2
import os
from extract_utils import face_alignment, convert_directory_to_image_file, delete_folder
from tqdm import tqdm
import numpy as np
from pathlib import Path
from face_aligner import FaceAligner
import natsort
import glob

def extract_frames_openface(dataset_folder_path='dataset/train/',
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


def extract_frames_realign(dataset_folder_path='dataset/train/', aligned_path='dataset/pipnet_align/train/'):
    # todo: format xxxx/xxxx_aligned/.jpg
    filenamePadding = 5
    if not os.path.exists(aligned_path):
        os.mkdir(aligned_path)
    directory = os.fsencode(dataset_folder_path+'mp4/')
    fa = FaceAligner(batch_size=16, pipnet=True)
    if not os.path.exists(aligned_path):
        os.mkdir(aligned_path)
        already_saved = []
    else:
        already_saved = natsort.natsorted(glob.glob(aligned_path+'/*'))
        already_saved = [x.split('/')[-1][:-4] for x in already_saved]

    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        folder = filename[:-4]
        if folder in already_saved:
            continue
        if filename.endswith(".mp4"):
            vid = cv2.VideoCapture(dataset_folder_path + 'mp4/' + filename)
            index = 0
            images = []
            names = []
            while (True):
                ret, frame = vid.read()
                if not ret:
                    break
                #H, W, _ = frame.shape
                #frame = cv2.resize(frame, (int(W / 2), int(H / 2)))
                images.append(frame)
                indexStr = 'frame_det_00_' + '0' * (filenamePadding - len(str(index))) + str(index) + '.jpg'
                names.append(indexStr)
                index += 1
            images = np.stack(images)#[:10]
            #names = names[:10]
            output_images, output_names = fa.alignFaceFromImages(images, names)
            folder_dir = aligned_path + folder + '/'
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            folder_dir = aligned_path + folder + '/' + folder + '_aligned/'
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            if len(output_names) == 0:
                continue
            for i in range(len(output_names)):
                cv2.imwrite(folder_dir+output_names[i], output_images[i])
        else:
            continue


if __name__ == "__main__":
    extract_frames_realign(dataset_folder_path='./dataset/train/', aligned_path='./dataset/pipnet_align/train/')
    #extract_frames_realign(dataset_folder_path='./dataset/val/')
    # /home/yini/OpenFace/build/bin/FeatureExtraction
    # /data/abaw5/val/
