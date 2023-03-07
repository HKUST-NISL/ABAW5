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


def get_of(final_images, k, face_pose_predictor, face_detector, optical_flow,
           gpu_frame, gpu_frame2, useGpu, ld):
    OFF_video = []
    for img_count in range(final_images.shape[0] - k):
        img1 = final_images[img_count]
        img2 = final_images[img_count + k]
        if (img_count == 0):
            reference_img = img1
            #detect = face_detector(reference_img, 1)
            num_of_face = ld.check_faces(reference_img)
            next_img = 0  # Loop through the frames until all the landmark is detected
            while (num_of_face == 0) and (img_count+next_img)<len(final_images)-1:
                next_img += 1
                reference_img = final_images[img_count + next_img]
                num_of_face = ld.check_faces(reference_img)
            #shape = face_pose_predictor(reference_img, detect[0])
            shape = ld.get_landmarks(reference_img)

            # Left Eye
            x11 = max(shape[36][0] - 15, 0) #max(shape.part(36).x - 15, 0)
            y11 = shape[36][1]
            x12 = shape[37][0]
            y12 = max(shape[37][1] - 15, 0)
            x13 = shape[38][0]
            y13 = max(shape[38][1] - 15, 0)
            x14 = min(shape[39][0] + 15, 128)
            y14 = shape[39][1]
            x15 = shape[40][0]
            y15 = min(shape[40][1] + 15, 128)
            x16 = shape[41][0]
            y16 = min(shape[41][1] + 15, 128)

            # Right Eye
            x21 = max(shape[42][0] - 15, 0)
            y21 = shape[42][1]
            x22 = shape[43][0]
            y22 = max(shape[43][1] - 15, 0)
            x23 = shape[44][0]
            y23 = max(shape[44][1] - 15, 0)
            x24 = min(shape[45][0] + 15, 128)
            y24 = shape[45][1]
            x25 = shape[46][0]
            y25 = min(shape[46][1] + 15, 128)
            x26 = shape[47][0]
            y26 = min(shape[47][1] + 15, 128)

            # ROI 1 (Left Eyebrow)
            x31 = max(shape[17][0] - 12, 0)
            y32 = max(shape[19][1] - 12, 0)
            #x33 = min(shape.part(21).x + 12, 128)
            y34 = min(shape[41][1] + 12, 128)

            # ROI 2 (Right Eyebrow)
            #x41 = max(shape.part(22).x - 12, 0)
            y42 = max(shape[24][1] - 12, 0)
            x43 = min(shape[26][0] + 12, 128)
            y44 = min(shape[46][1] + 12, 128)

            # ROI 3 #Mouth
            x51 = max(shape[60][0] - 12, 0)
            y52 = max(shape[50][1] - 12, 0)
            x53 = min(shape[64][0] + 12, 128)
            y54 = min(shape[57][1] + 12, 128)

            # Nose landmark
            x61 = shape[28][0]
            y61 = shape[28][1]

            '''for i in range(68):
                x, y = shape[i]
                cv2.circle(reference_img, (x, y), 1, (0, 0, 255), 2)
            cv2.imshow('1', reference_img)
            cv2.waitKey(0)'''

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

    imgPath = 'dataset/train/aligned/02770/02770_aligned/frame_det_00_000032.jpg'
    #imgPath = 'dataset/train/aligned/02767/02767_aligned/frame_det_00_000002.jpg'
    img = cv2.imread(imgPath, 1)
    img = cv2.resize(img, (128, 128))

    from PIPNet.lib.landmark_detection import LandmarkDetection
    ld = LandmarkDetection()
    num_of_face = ld.check_faces(img)
    ldmk_pipnet = ld.get_landmarks(img)

    #from PIPNet.lib.test import demo_image
    #ldmk_pipnet = demo_image(imgPath)

    '''detect = face_detector(img, 1)
    # face = dlib.rectangle(left=0, top=0, right=224, bottom=224)
    shape = face_pose_predictor(img, detect[0])
    ldmk_dlib = []
    for n in range(0, 68):
        x = shape.part(n).x
        y = shape.part(n).y
        ldmk_dlib.append((x, y))'''
    #img = cv2.resize(img, (224, 224))
    for i in range(68):
        x, y = ldmk_pipnet[i]
        cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
    #cv2.imwrite('images/dlib_sample.jpg', img)
    cv2.imshow('1', img)
    cv2.waitKey(0)

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
    compareDlibAndOpenface()
    #testOpticalFlow('/data/abaw5/train/aligned/00022/00022_aligned/frame_det_00_000001.jpg',
    #                '/data/abaw5/train/aligned/00022/00022_aligned/frame_det_00_000002.jpg')
