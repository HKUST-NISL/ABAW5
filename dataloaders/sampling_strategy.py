import random

# choice 1: average the output scores of all the frames in the video
# choice 2: randomly subsequence sampling
# choice 3: use the MaE scores to sample several fixed-length intervals

class SamplingStrategy():
    def __init__(self, sampling_choice=1):
        self.sampling_choice = sampling_choice

    def get_sampled_images(self, filenames):
        if self.sampling_choice == 1:
            imgRandomLen = 1
            random_indexes = random.randint(0, len(filenames) - imgRandomLen)
            random_images = filenames[random_indexes:(random_indexes + imgRandomLen)]
        return random_images