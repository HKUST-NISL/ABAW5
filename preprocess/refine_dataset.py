import cv2
import os
import natsort
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

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


def checkAllBlack(data_path):
    files = natsort.natsorted(glob.glob(data_path + "aligned/*"))
    allBlack=0
    allImages=0
    names = []
    for i in tqdm(range(len(files))):
        dir_sub = files[i]
        folder = dir_sub.split('/')[-1]
        image_path = dir_sub + '/' + folder + '_aligned'
        imageFiles = natsort.natsorted(glob.glob(image_path + "/frame*.jpg"))
        for dir_sub_vid_img in imageFiles:
            allImages += 1
            image = cv2.imread(dir_sub_vid_img, 1)
            if image.sum() == 0:
                allBlack += 1
                names.append(dir_sub_vid_img)
    df = pd.DataFrame(names, names)
    df.to_csv(data_path + 'blackImages.csv')
    print('black images: ', allBlack, allBlack/allImages)

if __name__ == '__main__':
    checkAllBlack('dataset/val/')
    #saveOpticalFlowScores('dataset/optical_flow/train/', 'dataset/train/', False)
    #saveOpticalFlowScores('/data/abaw5/optical_flow/train/', '/data/abaw5/train/', True)
    #saveOpticalFlowScores('/data/abaw5/optical_flow/val/', '/data/abaw5/val/', True)