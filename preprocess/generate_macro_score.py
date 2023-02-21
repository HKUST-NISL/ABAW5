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


def spotting(result, k, p):
    score_plot = np.array(result)
    score_plot_agg = score_plot.copy()
    for x in range(len(score_plot[k:-k])):
        score_plot_agg[x + k] = score_plot[x:x + 2 * k].mean()
    score_plot_agg = score_plot_agg[k:-k]
    threshold = score_plot_agg.mean() + p * (
            max(score_plot_agg) - score_plot_agg.mean())  # Moilanen threshold technique
    peaks, _ = find_peaks(score_plot_agg[:, 0], height=threshold[0], distance=k)
    # [peaks-k, peaks+k]
    # TODO: map back
    print(peaks)


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


def testing(model_path, data_path, save):
    predictor_model = "dataset/optical_flow/shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    batch_size = 10
    k = 18 # for macro and CASME
    model = SOFTNet()
    model.load_weights(model_path)
    save_path = data_path + '/optical_flow'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for feature in glob.glob(data_path + '/processed/*.npy'):
        data = np.load(feature) #[:50]
        flow_vectors = get_of(data, k, face_pose_predictor, face_detector) # 44, 42, 42, 3
        y = np.ones((data.shape[0]))
        result = model.predict_generator(
                generator(flow_vectors, y, batch_size),
                steps=len(flow_vectors)/batch_size,
                verbose=1
            )
        #print(result)
        if save:
            np.save(save_path+'/'+str(feature.split('/')[-1]), result)


def plot(path):
    k = 18
    p = 0.1 # only 63/137
    result = np.load(path)
    spotting(result, k, p)
    plt.plot(result)
    plt.show()


if __name__ == '__main__':
    '''testing('dataset/optical_flow/s1.hdf5',
            'dataset/val/',
            save=True)'''
    '''testing('dataset/optical_flow/s1.hdf5',
            'dataset/train/',
            save=True)'''
    plot('dataset/val/optical_flow/25066.npy')