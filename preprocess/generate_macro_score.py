import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import cv2
import random
import dlib
from get_optical_flow import get_of
import os
import glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
random.seed(1)
import natsort
from tqdm import tqdm


def spotting(result, k, p):
    score_plot = np.array(result)
    score_plot_agg = score_plot.copy()
    for x in range(len(score_plot[k:-k])):
        score_plot_agg[x + k] = score_plot[x:x + 2 * k].mean()
    score_plot_agg = score_plot_agg[:-k]
    threshold = score_plot_agg.mean() + p * (
            max(score_plot_agg) - score_plot_agg.mean())  # Moilanen threshold technique
    print('Threshold: ', threshold)
    peaks, _ = find_peaks(score_plot_agg[:, 0], height=threshold[0], distance=k)
    print('Peaks:', peaks/30+k/30)
    x = np.arange(score_plot_agg.shape[0]) / 30
    plt.plot(x, score_plot_agg+k/30)
    plt.xlabel('Sec')
    plt.ylabel('MaE spotting score')
    plt.show()


def normalize(images):
    for index in range(len(images)):
        for channel in range(3):
            images[index][:,:,channel] = cv2.normalize(images[index][:,:,channel], None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX)
    return images


def generator(X, y, batch_size=12, epochs=1):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            num_images = end - start
            X[start:end] = normalize(X[start:end])
            u = np.array(X[start:end])[:,:,:,0].reshape(num_images,42,42,1)
            v = np.array(X[start:end])[:,:,:,1].reshape(num_images,42,42,1)
            os = np.array(X[start:end])[:,:,:,2].reshape(num_images,42,42,1)
            yield [u, v, os], np.array(y[start:end])


def SOFTNet():
    inputs1 = layers.Input(shape=(42,42,1))
    conv1 = layers.Conv2D(3, (5,5), padding='same', activation='relu')(inputs1)
    pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv1)
    # channel 2
    inputs2 = layers.Input(shape=(42,42,1))
    conv2 = layers.Conv2D(5, (5,5), padding='same', activation='relu')(inputs2)
    pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv2)
    # channel 3
    inputs3 = layers.Input(shape=(42,42,1))
    conv3 = layers.Conv2D(8, (5,5), padding='same', activation='relu')(inputs3)
    pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv3)
    # merge
    merged = layers.Concatenate()([pool1, pool2, pool3])
    # interpretation
    merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(merged)
    flat = layers.Flatten()(merged_pool)
    dense = layers.Dense(400, activation='relu')(flat)
    outputs = layers.Dense(1, activation='linear')(dense)
    #Takes input u,v,s
    model = keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    sgd = keras.optimizers.SGD(lr=0.0005)
    model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model


def testing(model_path, data_path, save, batch):
    predictor_model = "dataset/MaE_model/shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    k = 18 # for macro and CASME
    model = SOFTNet()
    model.load_weights(model_path)
    save_path = data_path + '/MaE_score'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        already_saved = []
    else:
        already_saved = natsort.natsorted(glob.glob(save_path+'/*'))
        already_saved = [x.split('/')[-1][:-4] for x in already_saved]

    # Compute Optical Flow Features
    # optical_flow = cv2.DualTVL1OpticalFlow_create() #Depends on cv2 version
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    files = natsort.natsorted(glob.glob(data_path + "aligned/*"))
    for i in tqdm(range(len(files))):
        dir_sub = files[i]
        folder = dir_sub.split('/')[-1]
        if folder in already_saved:
            continue
        try:
            images = []
            image_path = dir_sub + '/' + folder + '_aligned'
            for dir_sub_vid_img in natsort.natsorted(glob.glob(image_path + "/frame*.jpg")):
                image = cv2.imread(dir_sub_vid_img, 0)  # 224, 224
                image = cv2.resize(image, (128, 128))
                images.append(image)
            images = np.stack(images)
            flow_vectors = get_of(images, k, face_pose_predictor, face_detector, optical_flow) # 44, 42, 42, 3
            y = np.ones((images.shape[0]))
            result = model.predict_generator(
                    generator(flow_vectors, y, batch),
                    steps=int(len(flow_vectors)/batch),
                    verbose=0
                )
            #print(result)
            if save:
                np.save(save_path+'/'+folder, result)
        except:
            print('Error when processing ', dir_sub)



def plot(path):
    k = 18
    p = 0
    result = np.load(path)
    spotting(result, k, p)


if __name__ == '__main__':
    testing('dataset/MaE_model/s1.hdf5',
            'dataset/val/',
            save=True, batch=10)
    testing('dataset/MaE_model/s1.hdf5',
            'dataset/train/',
            save=True, batch=10)
    #plot('dataset/train/MaE_score/08719.npy')