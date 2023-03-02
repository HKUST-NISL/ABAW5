import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import cv2
import csv

def pol2cart(rho, phi):  # Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def computeStrain(u, v):
    u_x = u - pd.DataFrame(u).shift(-1, axis=1)
    v_y = v - pd.DataFrame(v).shift(-1, axis=0)
    u_y = u - pd.DataFrame(u).shift(-1, axis=0)
    v_x = v - pd.DataFrame(v).shift(-1, axis=1)
    os = np.array(np.sqrt(u_x ** 2 + v_y ** 2 + 1 / 2 * (u_y + v_x) ** 2).ffill(1).ffill(0))
    return os


def get_of(final_images, k, face_pose_predictor, face_detector, optical_flow):
    OFF_video = []
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame2 = cv2.cuda_GpuMat()
    for img_count in range(final_images.shape[0] - k):
        img1 = final_images[img_count]
        img2 = final_images[img_count + k]
        gpu_frame.upload(img1)
        gpu_frame2.upload(img2)
        if (img_count == 0):
            reference_img = img1
            detect = face_detector(reference_img, 1)
            next_img = 0  # Loop through the frames until all the landmark is detected
            while (len(detect) == 0):
                next_img += 1
                reference_img = final_images[img_count + next_img]
                detect = face_detector(reference_img, 1)
            shape = face_pose_predictor(reference_img, detect[0])

            # Left Eye
            x11 = max(shape.part(36).x - 15, 0)
            y11 = shape.part(36).y
            x12 = shape.part(37).x
            y12 = max(shape.part(37).y - 15, 0)
            x13 = shape.part(38).x
            y13 = max(shape.part(38).y - 15, 0)
            x14 = min(shape.part(39).x + 15, 128)
            y14 = shape.part(39).y
            x15 = shape.part(40).x
            y15 = min(shape.part(40).y + 15, 128)
            x16 = shape.part(41).x
            y16 = min(shape.part(41).y + 15, 128)

            # Right Eye
            x21 = max(shape.part(42).x - 15, 0)
            y21 = shape.part(42).y
            x22 = shape.part(43).x
            y22 = max(shape.part(43).y - 15, 0)
            x23 = shape.part(44).x
            y23 = max(shape.part(44).y - 15, 0)
            x24 = min(shape.part(45).x + 15, 128)
            y24 = shape.part(45).y
            x25 = shape.part(46).x
            y25 = min(shape.part(46).y + 15, 128)
            x26 = shape.part(47).x
            y26 = min(shape.part(47).y + 15, 128)

            # ROI 1 (Left Eyebrow)
            x31 = max(shape.part(17).x - 12, 0)
            y32 = max(shape.part(19).y - 12, 0)
            x33 = min(shape.part(21).x + 12, 128)
            y34 = min(shape.part(41).y + 12, 128)

            # ROI 2 (Right Eyebrow)
            x41 = max(shape.part(22).x - 12, 0)
            y42 = max(shape.part(24).y - 12, 0)
            x43 = min(shape.part(26).x + 12, 128)
            y44 = min(shape.part(46).y + 12, 128)

            # ROI 3 #Mouth
            x51 = max(shape.part(60).x - 12, 0)
            y52 = max(shape.part(50).y - 12, 0)
            x53 = min(shape.part(64).x + 12, 128)
            y54 = min(shape.part(57).y + 12, 128)

            # Nose landmark
            x61 = shape.part(28).x
            y61 = shape.part(28).y

        flow = optical_flow.calc(gpu_frame, gpu_frame2, None)
        flow = flow.download()
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        u, v = pol2cart(magnitude, angle)
        os = computeStrain(u, v)

        # Features Concatenation into 128x128x3
        final = np.zeros((128, 128, 3))
        final[:, :, 0] = u
        final[:, :, 1] = v
        final[:, :, 2] = os

        # Remove global head movement by minus nose region
        final[:, :, 0] = abs(final[:, :, 0] - final[y61 - 5:y61 + 6, x61 - 5:x61 + 6, 0].mean())
        final[:, :, 1] = abs(final[:, :, 1] - final[y61 - 5:y61 + 6, x61 - 5:x61 + 6, 1].mean())
        final[:, :, 2] = final[:, :, 2] - final[y61 - 5:y61 + 6, x61 - 5:x61 + 6, 2].mean()

        # Eye masking
        left_eye = [(x11, y11), (x12, y12), (x13, y13), (x14, y14), (x15, y15), (x16, y16)]
        right_eye = [(x21, y21), (x22, y22), (x23, y23), (x24, y24), (x25, y25), (x26, y26)]
        cv2.fillPoly(final, [np.array(left_eye)], 0)
        cv2.fillPoly(final, [np.array(right_eye)], 0)

        # ROI Selection -> Image resampling into 42x22x3
        final_image = np.zeros((42, 42, 3))
        final_image[:21, :, :] = cv2.resize(final[min(y32, y42): max(y34, y44), x31:x43, :], (42, 21))
        final_image[21:42, :, :] = cv2.resize(final[y52:y54, x51:x53, :], (42, 21))
        OFF_video.append(final_image)
    OFF_video = np.stack(OFF_video)
    return OFF_video

def compareDlibAndOpenface():
    import dlib
    predictor_model = "dataset/MaE_model/shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    imgPath = 'dataset/train/aligned/02127/02127_aligned/frame_det_00_000001.jpg'
    img = cv2.imread(imgPath)
    detect = face_detector(img, 1)
    face = dlib.rectangle(left=0, top=0, right=224, bottom=224)
    shape = face_pose_predictor(img, face)
    ldmk_dlib = []
    for n in range(0, 68):
        x = shape.part(n).x
        y = shape.part(n).y
        ldmk_dlib.append((x, y))

    df = pd.read_csv("dataset/train/aligned/02127/02127.csv",dtype = {0:str}, nrows=1) #['x_0':'y_67']
    idx1 = df.columns.get_loc("x_0")
    idx2 = df.columns.get_loc("y_67")
    ldmk_openface = df.iloc[0, idx1:(idx2+1)].to_numpy().reshape(2, 68)

def testOpticalFlow(img1path, img2path):
    import time
    t1 = time.time()
    img1 = cv2.imread(img1path, 0)
    img2 = cv2.imread(img2path, 0)
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(img1, img2, None)
    t2 = time.time()
    print('cpu: ', str(t2-t1))

    t1 = time.time()
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(img1)
    gpu_frame2 = cv2.cuda_GpuMat()
    gpu_frame2.upload(img2)
    of_gpu = cv2.cuda.OpticalFlowDual_TVL1_create()
    flow_gpu = of_gpu.calc(gpu_frame, gpu_frame2, None)
    t2 = time.time()
    print('gpu: ', str(t2 - t1))


if __name__ == '__main__':
    #compareDlibAndOpenface()
    testOpticalFlow('/data/abaw5/train/aligned/00022/00022_aligned/frame_det_00_000001.jpg',
                    '/data/abaw5/train/aligned/00022/00022_aligned/frame_det_00_000002.jpg')
