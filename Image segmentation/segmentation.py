import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.distance import cdist


class Segmentation:
    def __init__(self, img: str,
                 radius:int,
                 sigma: float,
                 s_pxls: int,
                 r_width: int,
                 r_height: int,
                 spatial: bool):
        """
        This class encapsulates all information and functions needed to segment an RGB image.
        image:    rgb image to be segmented.
        radius:   bandwidth radius.
        sigma:    covariance parameter.
        r_width:  resize original image.
        """
        self.image = img
        self.radius = radius
        self.sigma = sigma
        self.skip_pxls = s_pxls
        self.r_width = r_width
        self.r_height = r_height
        self.spatial = spatial

    def preprocess(self):
        """
        Imports the image from main.py and applies augmentation if asked.
        Transforms numpy array [height, width, n_channels]  -> [height * width, n_channels], with n_channels 3 or 5.
        """
        img = cv2.imread(str(self.image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.r_width != None:
            img = cv2.resize(img, (self.r_width, self.r_height))
            print(f"Resizing image to {img.shape} .")

        self.true_dims = img.shape
        if self.spatial:
            x, y = np.meshgrid(range(img.shape[0]), range(img.shape[1]))
            data = np.concatenate([img.reshape(-1, 3), y.reshape(-1, 1), x.reshape(-1, 1)], axis=1)
            self.orig_img = data
            print("Using spatial information.")
        else:
            data = img.reshape(-1, 3)
            self.orig_img = img

        if self.skip_pxls:
            print(f"Original image pixels {len(data)} calculating centroids for {len(data[::self.skip_pxls])} of them.")
        self.img = data[::self.skip_pxls]  # IF ALL GOES TO HELL ADD .reshape(-1, 3)

        return self.img, self.orig_img

    def gaussian_kernel(self, points, center):
        """
        The gaussian kernel leads to less vibrant colors.
        """
        dist = cdist([center], points)
        return np.exp(-(dist / (2 * self.sigma ** 2)))

    def meanshift(self):
        """
        Initializing all centroids to the image pixels, applying the decision boundary, find the density mean
        and finally shift the corresponding centroid to that point.
        """
        self.centroids = self.img

        prev_centroids = self.centroids
        for i in tqdm(range(500)):
            for idx, cen in enumerate(self.centroids):
                dists = cdist([cen], self.img, metric="euclidean")

                # Find all the points that lie inside the circle radius.
                in_circle = (dists <= self.radius).T
                in_circle = in_circle.reshape(len(in_circle), )
                p_circle = self.img[in_circle]

                weights = self.gaussian_kernel(p_circle, cen)

                Mx = np.sum(weights.dot(p_circle[:, 0])) / np.sum(weights)
                My = np.sum(weights.dot(p_circle[:, 1])) / np.sum(weights)
                Mz = np.sum(weights.dot(p_circle[:, 2])) / np.sum(weights)
                if self.spatial:
                    Mxx = np.sum(weights.dot(p_circle[:, 3])) / np.sum(weights)
                    Myy = np.sum(weights.dot(p_circle[:, 4])) / np.sum(weights)

                if self.spatial:
                    self.centroids[idx] = np.array([Mx, My, Mz, Mxx, Myy])
                else:
                    self.centroids[idx] = np.array([Mx, My, Mz])
        return self.centroids

    def reconstruct(self):
        """
        Reconcstructing the image from centroids.
        Takes a column of pixel values and assigns the reconstructed pixel to least distance of the current image pixel
        vs all centroids..
        """
        temp_pxls = np.zeros(self.true_dims).reshape(-1, 3)
        labels = np.zeros_like(self.orig_img)
        if self.spatial:
            temp_img = self.orig_img.reshape(-1, 5)
            img_itr = self.orig_img.reshape(-1, 5)
        else:
            temp_img = self.orig_img.reshape(-1, 3)
            img_itr = self.orig_img.reshape(-1, 3)

        for i in tqdm(range(img_itr.shape[0])):
            d = cdist([temp_img[i]], self.centroids).flatten()
            labels[i] = np.where(d == d.min())[0][0]
            labels = labels.flatten()

            temp_pxls[i] = self.centroids[int(labels[i])][:3]

        self.img = temp_pxls.reshape(self.true_dims)
        self.img = self.img.astype(np.uint8)
        return self.img

