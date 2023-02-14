import subprocess
import glob
import os
import natsort

def crop_images(dataset_folder_path='./dataset/val/',
                fe_path='/home/eelassanai/Workspace/openface/OpenFace/build/bin/FeatureExtraction'):
    aligned_path = dataset_folder_path + 'aligned/'
    if not os.path.exists(aligned_path):
        os.mkdir(aligned_path)

    for subjectName in glob.glob(dataset_folder_path + 'images/*'):
        dataset_rawpic = subjectName

        # Create new directory for each subject
        dir_crop_sub = aligned_path + str(subjectName.split('/')[-1]) + '/'
        if not os.path.exists(dir_crop_sub):
            #shutil.rmtree(dir_crop_sub)
            os.mkdir(dir_crop_sub)
        print('Subject', subjectName.split('/')[-1])
        subprocess.run([fe_path, "-fdir", dataset_rawpic, "-out_dir", dir_crop_sub, "-nomask"])


if __name__ == '__main__':
    #fe_path: path to installed openFace feature extraction
    crop_images(dataset_folder_path='./dataset/val/', fe_path='')
