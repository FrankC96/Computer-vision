import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import datetime
import argparse


class Segmentation:
    def __init__(self, img: str,
                 radius:int,
                 sigma: float,
                 s_pxls: int,
                 r_width: int,
                 r_height: int):
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

    def preprocess(self):
        img = cv2.imread(str(self.image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.r_width != None:
            img = cv2.resize(img, (self.r_width, self.r_height))
            print(f"Resizing image to {img.shape} .")
        self.orig_img = img
        self.img = img[::self.skip_pxls].reshape(-1, 3)

        return self.img, self.orig_img

    def gaussian_kernel(self, points, center):
        """
        The gaussian kernel leads to less vibrant colors.
        """
        dist = cdist([center], points)

        return np.exp(-(dist / (2 * self.sigma ** 2)))

    def meanshift(self):
        self.centroids = self.img

        for i in tqdm(range(1000)):
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

                self.centroids[idx] = np.array([Mx, My, Mz])

        return self.centroids

    def reconstruct(self):
        temp_pxls = np.zeros_like(self.orig_img.reshape(-1, 3))
        labels = np.zeros_like(self.orig_img)
        temp_img = self.orig_img.reshape(-1, 3)
        for i in range(self.orig_img.reshape(-1, 3).shape[0]):
            d = cdist([temp_img[i]], self.centroids).flatten()
            labels[i] = np.where(d == d.min())[0][0]
            labels = labels.flatten()
            temp_pxls[i] = self.centroids[int(labels[i])]

        self.img = temp_pxls.reshape([self.orig_img.shape[0], self.orig_img.shape[1], 3])

        return self.img

