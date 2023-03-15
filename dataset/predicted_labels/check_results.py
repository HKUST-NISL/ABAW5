import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

labels_train = np.load('labels_train.npy')
labels_val = np.load('labels_val.npy')
preds_train = np.load('preds_train.npy')
preds_val = np.load('preds_val.npy')
vid_train = np.load('vids_train.npy')
vid_val = np.load('vids_val.npy')

def pcc(preds, labels):
    preds = np.mean(preds.reshape(-1, 1, 7), axis=1)
    labels = np.mean(labels.reshape(-1, 1, 7), axis=1)

    preds_mean = np.mean(preds, axis=0, keepdims=True)
    labels_mean = np.mean(labels, axis=0, keepdims=True)
    #top = np.sum((preds - preds_mean) * (labels - labels_mean), axis=0)
    top = (preds - preds_mean) * (labels - labels_mean)
    bottom = ((np.sum((preds - preds_mean) ** 2, axis=0) * np.sum((labels - labels_mean) ** 2,
                                                                        axis=0)) ** 0.5)
    pcc = top / bottom
    return pcc #np.mean(pcc)

def check_worst_videos():
    csv_file = '../data_info.csv'
    df = pd.read_csv(csv_file)
    columns = ['PCCloss','Adoration','Amusement','Anxiety','Disgust','Empathic-Pain','Fear','Surprise',
               'Adoration_pred','Amusement_pred','Anxiety_pred','Disgust_pred','Empathic-Pain_pred','Fear_pred','Surprise_pred']

    topn = 10
    values = []
    names = []
    pcc_loss_val = pcc(preds_val, labels_val).mean(axis=1) #.sum()
    pcc_loss_val_index = np.argsort(pcc_loss_val)[:topn]
    val_values = pcc_loss_val[pcc_loss_val_index]
    val_vids = vid_val[pcc_loss_val_index]

    for i in range(topn):
        pcc_loss = val_values[i]
        video_name = val_vids[i]
        loc = df['File_ID'] == '[' + str(video_name) + ']'
        info = df[loc]
        gt = info.iloc[0, 2:9].tolist()
        names.append(video_name)
        va = []
        va.append(pcc_loss)
        va.extend(gt)
        index_val = pcc_loss_val_index[i]
        preds = preds_val[index_val].tolist()
        va.extend(preds)
        values.append(va)

    df_save = pd.DataFrame(values, names, columns)
    df_save.to_csv('val_results.csv')

    values = []
    names = []
    pcc_loss_train = pcc(preds_train, labels_train).mean(axis=1) #.sum()
    pcc_loss_train_index = np.argsort(pcc_loss_train)[:topn]
    train_values = pcc_loss_train[pcc_loss_train_index]
    train_vids = vid_train[pcc_loss_train_index]
    for i in range(topn):
        pcc_loss = train_values[i]
        video_name = str(train_vids[i])
        if len(video_name) < 5:
            video_name = '0'*(5-len(video_name))+video_name
        loc = df['File_ID'] == '[' + str(video_name) + ']'
        info = df[loc]
        gt = info.iloc[0, 2:9].tolist()
        names.append(video_name)
        va = []
        va.append(pcc_loss)
        va.extend(gt)
        index_train = pcc_loss_train_index[i]
        preds = preds_train[index_train].tolist()
        va.extend(preds)
        values.append(va)

    df_save = pd.DataFrame(values, names, columns)
    df_save.to_csv('train_results.csv')

    '''plt.plot(pcc_loss_val)
    plt.title('PCC Loss - val')
    plt.show()'''

def visualizePCC7():
    names = ['Adoration','Amusement','Anxiety','Disgust','Empathic-Pain','Fear','Surprise']
    for i in range(7):
        #i=0
        plt.figure(i)
        name = names[i]
        labels = labels_val[:,i] #.flatten()
        preds = preds_val[:,i] #.flatten()
        labels = (labels - labels.mean())/labels.std()
        preds = (preds - preds.mean()) / preds.std()
        #res = labels * preds

        plt.scatter(labels, preds, alpha=0.05)
        plt.ylim(-5, 5)
        plt.grid()

        '''plt.hist(labels, bins=20, alpha = 0.5)
        plt.hist(preds, bins=20, alpha = 0.5)
        plt.ylim(0, 1800)'''

        plt.xlim(-5, 5)
        plt.title('val pcc visualization - '+name)
        plt.xlabel('Ground truth')
        plt.ylabel('Predicted')
        #plt.show()
        plt.savefig('pics/pcc-'+str(name))


def visualizePCC():
    names = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']

    labels = labels_val.flatten()
    preds = preds_val.flatten()
    labels = (labels - labels.mean()) / labels.std()
    preds = (preds - preds.mean()) / preds.std()

    plt.scatter(labels, preds, alpha=0.1/7)
    plt.ylim(-3, 3)
    plt.grid()

    # plt.hist(labels, bins=20, alpha = 0.5)
    # plt.hist(preds, bins=20, alpha = 0.5)

    plt.xlim(-5, 5)
    plt.title('val pcc visualization - all')
    plt.xlabel('Ground truth')
    plt.ylabel('Predicted')
    #plt.show()
    plt.savefig('pics/hist-pcc-all')


def showImages(csvFile='train_results.csv', datasetFile='dataset/train/aligned/'):
    df = pd.read_csv(csvFile)
    fileNames = df[df.columns[0]].tolist()
    imgs = []
    for vid in fileNames:
        imgPath = datasetFile + str(vid) + '/' + str(vid) + '_aligned/' + 'frame_det_00_000005.jpg'
        img = cv2.imread(imgPath, 1)
        img = img[..., ::-1]
        imgs.append(img)
    f, axarr = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            axarr[i, j].imshow(imgs[i*5+j])
            axarr[i, j].set_title(fileNames[i*5+j])
            axarr[i, j].set_axis_off()
    plt.show()
    plt.savefig('./worst_performance_train')

if __name__ == '__main__':
    visualizePCC7()
    #check_worst_videos()
    #showImages()