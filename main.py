import datetime
import argparse
import matplotlib.pyplot as plt

from segmentation import Segmentation

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(description='Image segmentation with mean shift clustering')

parser.add_argument('img', type=str, help='Path to image file.')
parser.add_argument('radius', type=int, help='Bandwidth radius.')
parser.add_argument('--sigma', type=float, help='Gaussian kernel covariance parameter.', default=15)
parser.add_argument('--s_pxls', type=int, help='Pixels to skip in image.', default=500)
parser.add_argument('--r_width', type=int, help='Image width resize.', default=None)
parser.add_argument('--r_height', type=int, help='Image height resize.', default=None)

args = parser.parse_args()

arg0_value = args.img
arg1_value = args.radius
arg2_value = args.sigma
arg3_value = args.s_pxls
arg4_value = args.r_width
arg5_value = args.r_height


s = Segmentation(arg0_value, arg1_value, arg2_value, arg3_value, arg4_value, arg5_value)

s.preprocess()
s.meanshift()
seg_img = s.reconstruct()

plt.imshow(seg_img)
plt.imsave(f"./out/{timestamp}_R{s.radius}_S{s.sigma}.png", seg_img)
plt.show()
