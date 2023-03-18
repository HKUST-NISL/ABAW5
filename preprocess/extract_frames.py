import cv2
import os
from extract_utils import face_alignment, convert_directory_to_image_file, delete_folder
from tqdm import tqdm
import numpy as np
from pathlib import Path
from face_aligner import FaceAligner
import natsort
import glob
import pandas as pd

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


def extract_frames_realign(dataset_folder_path='dataset/train/', aligned_path='dataset/pipnet_align/train/',
                           ldmk_path='./data/pipnet_align/landmarks/train/'):
    # todo: format xxxx/xxxx_aligned/.jpg
    filenamePadding = 5
    if not os.path.exists(aligned_path):
        os.mkdir(aligned_path)
    if not os.path.exists(ldmk_path):
        os.mkdir(ldmk_path)
    directory = os.fsencode(dataset_folder_path+'mp4/')
    fa = FaceAligner(batch_size=16, pipnet=True)
    if not os.path.exists(aligned_path):
        os.mkdir(aligned_path)
        already_saved = []
    else:
        already_saved = natsort.natsorted(glob.glob(aligned_path+'/*'))
        already_saved = [x.split('/')[-1] for x in already_saved]

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
            images = np.stack(images) #[:10] #todo: delete
            #names = names[:10]
            output_images, output_names, output_landmarks = fa.alignFaceFromImages(images, names)
            folder_dir = aligned_path + folder + '/'
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)

            if len(output_landmarks) != 0:
                H, W, _ = images[0].shape
                output_landmarks = np.array(output_landmarks, dtype=float)
                output_landmarks[:,:,0] /= float(W)
                output_landmarks[:,:,1] /= float(H)
                output_landmarks[:, :, [0, 1]] = output_landmarks[:, :, [1, 0]]

                df = pd.DataFrame(output_landmarks.reshape(-1, 68*2), output_names) #68,2 -> flatten
                df.to_csv(ldmk_path + '/' + folder + '.csv')

            folder_dir = aligned_path + folder + '/' + folder + '_aligned/'
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
            if len(output_names) == 0:
                continue
            for i in range(len(output_names)):
                cv2.imwrite(folder_dir+output_names[i], output_images[i])
        else:
            continue


def get_video_information(dataset_folder_path='dataset/train/',
                           ldmk_path='./data/pipnet_align/landmarks/train/',
                          csv_name='./dataset/pipnet_align/train_video_info.csv'):
    # todo: save fps, height, width, median eye distance
    vids = []
    video_information = []
    directory = os.fsencode(dataset_folder_path + 'mp4/')

    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        folder = filename[:-4]
        vids.append(folder)
        videoEntry = [] # order: FPS, height, width, median eye distance

        if filename.endswith(".mp4"):
            vidCap = cv2.VideoCapture(dataset_folder_path + 'mp4/' + filename)
            fps = vidCap.get(cv2.CAP_PROP_FPS)
            videoEntry.append(round(fps))
            ret, frame = vidCap.read()
            if not ret:
                break
            H, W, _ = frame.shape
            videoEntry.append(H)
            videoEntry.append(W)

            ldmk_csv = ldmk_path + folder + '.csv'
            try:
                df = pd.read_csv(ldmk_csv)
            except:
                videoEntry.append(np.nan)
                video_information.append(videoEntry)
                continue

            ldmk_values = df.values[:, 1:].reshape(-1, 68, 2)
            ldmk_values[:,:,0] *= H
            ldmk_values[:, :, 1] *= W  # 356, 68, 2

            lm_eye_left = ldmk_values[:,36: 42] # 356, 6, 2
            lm_eye_right = ldmk_values[:,42: 48]
            eye_left = np.mean(lm_eye_left, axis=1)
            eye_right = np.mean(lm_eye_right, axis=1)
            eye_to_eye = eye_right - eye_left  # 356, 2
            distances = (eye_to_eye[:,0]**2+eye_to_eye[:,1]**2)**0.5
            distance = np.median(distances)
            videoEntry.append(distance)
            video_information.append(videoEntry)

    df = pd.DataFrame(video_information, vids, columns=['FPS', 'height', 'width', 'eye distance'])
    df.to_csv('./dataset/pipnet_align/train_video_info.csv')


if __name__ == "__main__":
    get_video_information(dataset_folder_path='/Users/adia/Desktop/abaw/datasets/train/',
                           ldmk_path='./dataset/pipnet_align/landmarks/train/',
                          csv_name='./dataset/pipnet_align/train_video_info.csv')
    '''extract_frames_realign(dataset_folder_path='/Users/adia/Desktop/abaw/datasets/val/',
                           aligned_path='./dataset/pipnet_align/val/',
                           ldmk_path='./dataset/pipnet_align/landmarks/val/')'''
    #extract_frames_realign(dataset_folder_path='./dataset/val/')
    # /home/yini/OpenFace/build/bin/FeatureExtraction
    # /data/abaw5/val/
