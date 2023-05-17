import datetime
import argparse
import cv2
import matplotlib.pyplot as plt

from segmentation import Segmentation

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(description='Image segmentation with mean shift clustering')

parser.add_argument('img', type=str, help='Path to image file.')
parser.add_argument('radius', type=int, help='Bandwidth radius.')
parser.add_argument('--sigma', type=float, help='Gaussian kernel covariance parameter.', default=40)
parser.add_argument('--s_pxls', type=int, help='Pixels to skip in image.', default=100)
parser.add_argument('--r_width', type=int, help='Image width resize.', default=None)
parser.add_argument('--r_height', type=int, help='Image height resize.', default=None)
parser.add_argument('--spatial', type=bool, help='Add spatial information to the algorithm.', default=False)
parser.add_argument('--blur', type=int, help='Blur the output image.', default=None)

args = parser.parse_args()

arg0_value = args.img
arg1_value = args.radius
arg2_value = args.sigma
arg3_value = args.s_pxls
arg4_value = args.r_width
arg5_value = args.r_height
arg6_value = args.spatial
arg7_value = args.blur

if arg7_value and (arg7_value % 2) == 0:
    raise ValueError("The kernel size parameter must be odd.")

s = Segmentation(arg0_value, arg1_value, arg2_value, arg3_value, arg4_value, arg5_value, arg6_value)


s.preprocess()
s.meanshift()
seg_img = s.reconstruct()

if arg7_value:
    print(f"Blurring image by a factor of {arg7_value}.")
    seg_img = cv2.medianBlur(seg_img, arg7_value)
    # seg_img = cv2.cvtColor(img_median, cv2.COLOR_BGR2RGB)

plt.imshow(seg_img)
plt.imsave(f"./out_imgs/{timestamp}_R{s.radius}_S{s.sigma}.png", seg_img)
plt.show()
