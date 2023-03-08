import random
import numpy as np
from scipy.signal import find_peaks

# choice 1: randomly sample a subset of frames
# choice 2: use the MaE scores to sample several fixed-length intervals

def peakDetection(score_plot, k, p):
    if score_plot.sum() == 0:
        return np.random.choice(np.arange(score_plot.shape[0]-k), 1, replace=False)
    else:
        score_plot_agg = score_plot.copy()
        for x in range(len(score_plot[k:-k])):
            score_plot_agg[x + k] = score_plot[x:x + 2 * k].mean()
        score_plot_agg = score_plot_agg[:-k]
        threshold = score_plot_agg.mean() + p * (
                max(score_plot_agg) - score_plot_agg.mean())  # Moilanen threshold technique
        try:
            peaks, _ = find_peaks(score_plot_agg[:, 0], height=threshold[0], distance=k)
            peak = np.random.choice(peaks, 1, replace=False)
            return peak
        except:
            return np.random.choice(np.arange(score_plot.shape[0] - k), 1, replace=False)


class SamplingStrategy:
    def __init__(self, dataset_folder_path, sampling_choice=1):
        self.sampling_choice = sampling_choice
        self.data_dir = dataset_folder_path
        self.k = 20

    def get_sampled_paths(self, image_paths, snippet_size):
        if self.sampling_choice == 1:
            sampled_paths = np.random.choice(image_paths, snippet_size, replace=False)
        elif self.sampling_choice == 2:
            length = len(image_paths)  # 144
            vid = image_paths[0].split('/')[-3]
            macro_score_path = self.data_dir + '/' + vid + '.npy'
            try:
                macro_score = np.load(macro_score_path)  # 126, 1
            except:
                macro_score = np.zeros(length)
            # peak detection
            peak = peakDetection(macro_score, k=self.k, p=0)[0]
            sampled_paths = image_paths[peak:(peak+2*self.k)]
        return sampled_paths