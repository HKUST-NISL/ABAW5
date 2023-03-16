# import the necessary packages
#from .helpers import FACIAL_LANDMARKS_IDXS
#from .helpers import shape_to_np
import numpy as np
import cv2
import face_alignment
import torch
import os
from PIL import Image
import scipy.ndimage
import PIL.Image
import time
from PIPNet.lib.landmark_detection import LandmarkDetection

device = "cuda" if torch.cuda.is_available() else "cpu"
class FaceAligner:
    def __init__(self, batch_size, pipnet, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=224, desiredFaceHeight=None):
        self.batch_size = batch_size
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
        self.usePipnet = pipnet
        #if self.usePipnet:
        self.pipnet = LandmarkDetection()
        #else:
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                               device=device)

    def getLdmkFromBatch(self, batch):
        preds = self.fa.get_landmarks_from_batch(batch)
        return preds

    def getLdmkFromDir(self, dir):
        files = natsort.natsorted(glob.glob(dir+'/frame*.jpg'))
        images = []
        names = []
        for file in files:
            image = cv2.imread(file)
            H, W, _ = image.shape
            image = cv2.resize(image, (int(W/2), int(H/2)))
            images.append(image)
            names.append(file.split('/')[-1])
        images = np.stack(images)[:3] #todo
        names = names[:10]
        if not self.usePipnet:
            torch_images = torch.from_numpy(images).to(device).permute(0, 3, 1, 2)
            output_landmarks = []
            for i in range(0, images.shape[0], self.batch_size):
                image = torch_images[i:(i+self.batch_size)]
                x = self.getLdmkFromBatch(image)
                output_landmarks.extend(x)
        else:
            output_landmarks = self.getaLdmkFromImagesPipnet(images)
        return images, output_landmarks, names

    def getLdmkFromImages(self, images):
        output_landmarks = []
        for i in range(0, images.shape[0], self.batch_size):
            torch_images = torch.from_numpy(images[i:(i+self.batch_size)]).to(device).permute(0, 3, 1, 2)
            x = self.getLdmkFromBatch(torch_images)
            output_landmarks.extend(x)
            del torch_images
        return output_landmarks

    def alignFaceFromDir(self, dir):
        images, ldmks, names = self.getLdmkFromDir(dir)
        output_images = []
        output_names = []
        for i in range(len(ldmks)):
            if ldmks[i] == []:
                continue
            cropped = self.alignFaceOneImage(images[i], ldmks[i])
            output_images.append(cropped)
            output_names.append(names[i])
        return output_images, output_names

    def getaLdmkFromImagesPipnet(self, images):
        ldmks = []
        for i in range(images.shape[0]):
            image = images[i]
            ldmk = self.pipnet.get_landmarks(image)
            ldmks.append(ldmk)
        return ldmks

    def alignFaceFromImages(self, images, names):
        if self.usePipnet:
            ldmks = self.getaLdmkFromImagesPipnet(images)
        else:
            ldmks = self.getLdmkFromImages(images)
        output_images = []
        output_names = []
        output_ldmk = []
        for i in range(len(ldmks)):
            if ldmks[i] == []:
                continue
            cropped = self.alignFaceOneImage(images[i], ldmks[i])
            output_images.append(cropped)
            output_names.append(names[i])
            output_ldmk.append(ldmks[i])
        return output_images, output_names, output_ldmk

    def alignFaceOneImage(self, img, face_landmarks, output_size=224, enable_padding=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        lm = np.array(face_landmarks)
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 1, np.hypot(*eye_to_mouth) * 0.9)  # 2.0, 1.8
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.3  # 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))

        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        '''pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]'''
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def align_face(self, img, output_size=224, transform_size=224, enable_padding=False):
        face_landmarks = self.fa.get_landmarks(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        lm = np.array(face_landmarks)
        '''lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down'''
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        #lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 1, np.hypot(*eye_to_mouth) * 0.9) #2.0, 1.8
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.3 #0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        '''a1 = crop[0]
        a2 = crop[2]
        b1 = crop[1]
        b2 = crop[3]
        k_a = (a2 - a1) / 2
        k_b = (b2 - b1) / 2
        ratio = 0.3
        crop = (a1+ratio*k_a, b1+ratio*k_b, a2-ratio*k_a, b2-ratio*k_b)'''
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        '''img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)'''
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def get_info(self, img):
        face_landmarks = self.fa.get_landmarks(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        lm = np.array(face_landmarks)
        '''lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down'''
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        #lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        print('eye to eye, ', eye_to_eye)
        print('eye to mouth, ', eye_to_mouth)

if __name__ == '__main__':
    import natsort
    import glob
    import torch
    a = FaceAligner(batch_size=2, pipnet=False)
    dir = 'dataset/train/images/00062/'
    output_images, output_names = a.alignFaceFromDir(dir)
    save = 'dataset/train/re_aligned/02127_2/'
    for i in range(len(output_names)):
        image = output_images[i]
        name = output_names[i]
        cv2.imwrite(save+name, image)
