import cv2
import os

def extract_frames(dataset_folder_path='dataset/train/'):
    filenamePadding = 5
    directory = os.fsencode(dataset_folder_path+'mp4/')
    if not os.path.exists(dataset_folder_path + 'images'):
        os.makedirs(dataset_folder_path + 'images')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
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
                print('Creating...' + name)
                cv2.imwrite(name, frame)
                index += 1
        else:
            continue


if __name__ == "__main__":
    extract_frames(dataset_folder_path='./dataset/train/')
    extract_frames(dataset_folder_path='./dataset/val/')
