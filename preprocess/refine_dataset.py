import cv2
import os
import natsort
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
#from PIPNet.FaceBoxesV2.faceboxes_detector import FaceBoxesDetector
from face_aligner import FaceAligner
from PIL import Image
import matplotlib.pyplot as plt
import subprocess

# 1. Contrast Limited Adaptive Histogram Equalization
# 2. detect rotated images
# 3. detect bad quality videos
# 4. calculate optical flow scores

# 1.
'''import numpy as np
import cv2
img = cv2.imread('tsukuba_l.png',0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv2.imwrite('clahe_2.jpg',cl1)'''


def saveOpticalFlowScores(save_path, data_path, useGpu):
    k=1
    if useGpu:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame2 = cv2.cuda_GpuMat()
    else:
        gpu_frame = gpu_frame2 = None
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            already_saved = []
        else:
            already_saved = natsort.natsorted(glob.glob(save_path + '/*'))
            already_saved = [x.split('/')[-1][:-4] for x in already_saved]

    if useGpu:
        optical_flow = cv2.cuda.OpticalFlowDual_TVL1_create()
    else:
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    files = natsort.natsorted(glob.glob(data_path + "aligned/*"))
    for i in tqdm(range(len(files))):
        names = []
        values = []
        dir_sub = files[i]
        folder = dir_sub.split('/')[-1]
        if folder in already_saved:
            print('skip', folder)
            continue

        images = []
        image_path = dir_sub + '/' + folder + '_aligned'
        imageFiles = natsort.natsorted(glob.glob(image_path + "/frame*.jpg"))
        for dir_sub_vid_img in imageFiles:
            image = cv2.imread(dir_sub_vid_img, 1)  # 224, 224
            image = cv2.resize(image, (128, 128))
            images.append(image)
        final_images = np.stack(images)
        # imageFiles[0], 0
        names.append(imageFiles[0].split('/')[-1])
        values.append(0)
        for img_count in range(final_images.shape[0] - k):
            img1 = final_images[img_count]
            img2 = final_images[img_count + k]
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            if useGpu:
                gpu_frame.upload(img1)
                gpu_frame2.upload(img2)
                flow = optical_flow.calc(gpu_frame, gpu_frame2, None)
                flow = flow.download()
            else:
                flow = optical_flow.calc(img1, img2, None)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            result = abs(magnitude).mean()
            names.append(imageFiles[img_count].split('/')[-1])
            values.append(result)

        if save_path is not None:
            df = pd.DataFrame(values, names)
            df.to_csv(save_path + '/' + folder + '.csv')


def checkAllBlack(data_path, saveName):
    files = natsort.natsorted(glob.glob(data_path + "aligned/*"))
    allBlack=0
    allImages=0
    names = []
    videos = []
    videoNumber = 0
    for i in tqdm(range(len(files))):
        dir_sub = files[i]
        folder = dir_sub.split('/')[-1]
        image_path = dir_sub + '/' + folder + '_aligned'
        imageFiles = natsort.natsorted(glob.glob(image_path + "/frame*.jpg"))
        isThisVideo = 0
        for dir_sub_vid_img in imageFiles:
            allImages += 1
            image = cv2.imread(dir_sub_vid_img, 1)
            if image.sum() == 0:
                if isThisVideo == 0:
                    isThisVideo = 1
                allBlack += 1
                names.append(dir_sub_vid_img)
                videos.append(folder)
        videoNumber += isThisVideo
    df = pd.DataFrame(videos, names)
    df.to_csv(data_path + saveName + '.csv')
    print('# black frames: ', allBlack, allBlack/allImages)
    print('# black videos: ', videoNumber)

def checkAllBlackAfterRealign(blackImageCsv, data_path, saveName):
    df = pd.read_csv(blackImageCsv)
    realignedVideo = set(df.iloc[:, 1].tolist())

    files = natsort.natsorted(glob.glob(data_path + "aligned/*"))
    allBlack=0
    allImages=0
    names = []
    videos = []
    videoNumber = 0
    for i in tqdm(range(len(files))):
        dir_sub = files[i]
        folder = dir_sub.split('/')[-1]
        if int(folder) in realignedVideo:
            image_path = data_path + "re_aligned/" + folder + '/' + folder + '_aligned'
        else:
            image_path = dir_sub + '/' + folder + '_aligned'
        imageFiles = natsort.natsorted(glob.glob(image_path + "/frame*.jpg"))
        isThisVideo = 0
        for dir_sub_vid_img in imageFiles:
            allImages += 1
            image = cv2.imread(dir_sub_vid_img, 1)
            if image.sum() == 0:
                if isThisVideo == 0:
                    isThisVideo = 1
                allBlack += 1
                names.append(dir_sub_vid_img)
                videos.append(folder)
        videoNumber += isThisVideo
    df = pd.DataFrame(videos, names)
    df.to_csv(data_path + saveName + '.csv')
    print('# black frames: ', allBlack, allBlack/allImages)
    print('# black videos: ', videoNumber)


def deleteBlackImagesRealign(data_path):
    files = natsort.natsorted(glob.glob(data_path + "re_aligned/*"))
    totalDelete = 0
    for i in tqdm(range(len(files))):
        dir_sub = files[i]
        folder = dir_sub.split('/')[-1]
        image_path = dir_sub + '/' + folder + '_aligned'
        imageFiles = natsort.natsorted(glob.glob(image_path + "/frame*.jpg"))
        for dir_sub_vid_img in imageFiles:
            image = cv2.imread(dir_sub_vid_img, 1)
            if image.sum() == 0:
                subprocess.run(["rm", dir_sub_vid_img])
                totalDelete += 1
    print('total delete frame: ', totalDelete)


def reDetectFacesDrawExample(blackImageFile, savePath, videoPath):
    filenamePadding = 6
    df = pd.read_csv(blackImageFile)
    videosToRedetect = list(set(df.iloc[:,1].tolist()))[:10]
    face_aligner = FaceAligner(batch_size=2, pipnet=True)

    new_aligned_path = savePath + 're_aligned/'
    if not os.path.exists(new_aligned_path):
        os.mkdir(new_aligned_path)

    origins = []
    realigned = []
    fileNames = []

    for file in tqdm(videosToRedetect):
        fileNames.append(file)
        filename = '0'*(5-len(str(file)))+str(file)+'.mp4'
        if not os.path.exists(new_aligned_path+filename[:-4]):
            os.mkdir(new_aligned_path+filename[:-4])
        vid = cv2.VideoCapture(videoPath + filename)
        ret, frame_ori = vid.read()
        frame = face_aligner.align_face(frame_ori)

        ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        #ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb_img = clahe.apply(ycrcb_img)
        frame = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


        frame = frame[..., ::-1]
        #frame_ori = frame_ori[..., ::-1]
        #origins.append(frame_ori)
        realigned.append(frame)

    imgs = realigned
    f, axarr = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            axarr[i, j].imshow(imgs[i * 5 + j])
            axarr[i, j].set_title(fileNames[i * 5 + j])
            axarr[i, j].set_axis_off()
    plt.show()
    #plt.savefig('./realign-val-ori')


def reDetectFaces(blackImageFile, savePath, videoPath):
    filenamePadding = 6
    df = pd.read_csv(blackImageFile)
    videosToRedetect = set(df.iloc[:,1].tolist())
    face_aligner = FaceAligner()

    new_aligned_path = savePath  #+ 're_aligned/'
    if not os.path.exists(new_aligned_path):
        os.mkdir(new_aligned_path)

    for file in tqdm(videosToRedetect):
        filename = '0'*(5-len(str(file)))+str(file)+'.mp4'
        if not os.path.exists(new_aligned_path+filename[:-4]):
            os.mkdir(new_aligned_path+filename[:-4])
        final_dir = new_aligned_path+filename[:-4] + '/'+filename[:-4]+'_aligned'
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)
        vid = cv2.VideoCapture(videoPath + filename)
        index = 0
        while True:
            ret, frame = vid.read()
            indexStr = '0' * (filenamePadding - len(str(index))) + str(index)
            name = final_dir + '/frame_det_00_' + indexStr + '.jpg'
            if not ret:
                break
            try:
                frame = face_aligner.align_face(frame)
                cv2.imwrite(name, frame)
                index += 1
            except:
                #size = frame.shape
                #frame = np.zeros((size))
                #print(filename, name)
                index += 1
                continue


def drawOpenFaceAligned():
    files = natsort.natsorted(glob.glob('dataset/train/aligned/*'))
    images = []
    names = []
    for i in tqdm(range(len(files))):
        dir_sub = files[i]
        folder = dir_sub.split('/')[-1]
        names.append(folder)
        image_path = dir_sub + '/' + folder + '_aligned'
        imageFiles = natsort.natsorted(glob.glob(image_path + "/frame*.jpg"))
        image = cv2.imread(imageFiles[0], 1)
        images.append(image)
    f, axarr = plt.subplots(2, 4)
    for i in range(2):
        for j in range(4):
            axarr[i, j].imshow(images[i * 4 + j])
            axarr[i, j].set_title(names[i * 4 + j])
            axarr[i, j].set_axis_off()
    plt.show()

def effectNetExample():
    files = natsort.natsorted(glob.glob('dataset/effectNetExample/*'))
    images = []
    names = []
    for i in tqdm(range(len(files))):
        image = cv2.imread(files[i], 1)
        image = image[..., ::-1]
        images.append(image)
        names.append(files[i].split('/')[-1])
    f, axarr = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            axarr[i, j].imshow(images[i * 5 + j])
            axarr[i, j].set_title(names[i * 5 + j])
            axarr[i, j].set_axis_off()
    plt.show()

def compareTwoAlignedFaces():
    face_aligner = FaceAligner()
    a1 = cv2.imread('dataset/original.jpg')
    aligned_n = face_aligner.align_face(a1)
    #cv2.imshow('0', aligned_n)
    #cv2.waitKey(0)
    aligned_e = cv2.imread('dataset/effectNetExample/24.jpg')
    print('our align')
    face_aligner.get_info(aligned_n)
    print('effectnet align')
    face_aligner.get_info(aligned_e)

def checkPipnetAlignedFrames(openface_aligned_dir, pipnet_aligned_dir,
                             landmark_dir):
    openface_files = natsort.natsorted(glob.glob(openface_aligned_dir+'*'))
    pipnet_files = natsort.natsorted(glob.glob(pipnet_aligned_dir+'*'))
    landmark_files = natsort.natsorted(glob.glob(landmark_dir+'*'))
    openface = []
    for o in openface_files:
        folder = o.split('/')[-1]
        x = len(natsort.natsorted(glob.glob(o+'/'+folder+'_aligned/*')))
        openface.append(x)
    pipnet = []
    for o in pipnet_files:
        folder = o.split('/')[-1]
        x = len(natsort.natsorted(glob.glob(o + '/' + folder + '_aligned/*')))
        pipnet.append(x)
    landmarks = []
    for o in landmark_files:
        df = pd.read_csv(o)
        x = len(df.index)
        landmarks.append(x)
    len_openface = len(openface)
    sum_openface = np.sum(openface)
    len_pipnet = len(pipnet)
    sum_pipnet = np.sum(pipnet)
    len_landmarks = len(landmarks)
    sum_landmarks = np.sum(landmarks)

    print('len openface ', len_openface)
    print('sum openface ', sum_openface)
    print('len pipnet ', len_pipnet)
    print('sum pipnet ', sum_pipnet)
    print('len landmarks ', len_landmarks)
    print('sum landmarks ', sum_landmarks)
    # calculate missing frames, invalid videos
    # one by one check landmarks
    print('missing frames ', sum_openface-sum_pipnet)
    pipnet = np.array(pipnet)
    x = np.where(pipnet == 0)
    print('invalid videos ', x[0].shape)


if __name__ == '__main__':
    checkPipnetAlignedFrames('dataset/train/aligned/', 'dataset/pipnet_align/train/',
                             'dataset/pipnet_align/landmarks/train/')
    #compareTwoAlignedFaces()
    #effectNetExample()
    #reDetectFacesDrawExample('dataset/val/blackImages_before.csv', 'dataset/val/re_aligned2/', '/Users/adia/Desktop/abaw/datasets/val/mp4/')
    #saveOpticalFlowScores('dataset/optical_flow/train/', 'dataset/train/', False)
    #saveOpticalFlowScores('/data/abaw5/optical_flow/train/', '/data/abaw5/train/', True)
    #saveOpticalFlowScores('/data/abaw5/optical_flow/val/', '/data/abaw5/val/', True)
    #drawOpenFaceAligned()

    #checkAllBlackAfterRealign('dataset/train/blackImages_before.csv', 'dataset/train/', 'blackImages_realigned')
    # deleteBlackImagesRealign('dataset/train/')