import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.spatial.distance import cdist
from tqdm import tqdm


def gaussian_kernel(points, center, sigma):
    dist = cdist([center], points)

    return np.exp(-(dist / (2 * sigma**2)))

img = cv2.imread("tiger.jpg")

width = 50
height = 50

resized_img = cv2.resize(img, (width, height))
plt.figure(1)
# resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2LAB)
plt.imshow(resized_img)

X = resized_img.reshape(-1, 3)

center = np.zeros_like(X)
labels = np.zeros_like(X)
for i in range(len(X)):
    center[i] = X[i]


# Define the circle boundary
# r = 50
# simga = 5
sigma = 10
r = 50

M_x = []
M_y = []
M_z = []

for i in tqdm(range(200)):
    for idx, cen in enumerate(center):
        """
        It is possible to take just the mean of all the points inside the circle and shift the center,
        but this will result in a more "jagged" outcome. Instead you have to use the gaussian kernel.
        """
        dists = cdist([cen], X, metric="euclidean")

        # Find all the points that lie inside the circle radius.
        in_circle = (dists <= r).T
        in_circle = in_circle.reshape(len(in_circle), )
        p_circle = X[in_circle]

        weights = gaussian_kernel(p_circle, cen, sigma)

        Mx = np.sum(weights.dot(p_circle[:, 0])) / np.sum(weights)
        M_x.append(Mx)
        My = np.sum(weights.dot(p_circle[:, 1])) / np.sum(weights)
        M_y.append(My)
        Mz = np.sum(weights.dot(p_circle[:, 2])) / np.sum(weights)
        M_z.append(Mz)

        center[idx] = np.array([Mx, My, Mz])

ass_pxls = np.zeros_like(X)
for i in range(X.shape[0]):
    d = cdist([X[i]], center).flatten()
    labels[i] = np.where(d == d.min())[0][0]
    labels = labels.flatten()
    ass_pxls[i] = center[int(labels[i])]

r_img = ass_pxls.reshape([50, 50, 3])
plt.figure(2)

plt.imshow(r_img)
plt.imsave("segmentedIm.jpg")
plt.show()
