Image segmentation using the mean shift clustering algorithm.

Calling main.py  
Example usage:  
python main.py "test_imgs/tiger.jpg" 40 --s_pxls 100

| arg.       | description                                             |
|------------|---------------------------------------------------------|
| img        | Path to image file.                                     |
| radius     | Bandwidth radius.                                       |
| --sigma    | Gaussian kernel covariance parameter.  default=40       |
| --s_pxls   | Pixels to skip in image.               default=100      |
| --r_width  | Image width resize.                    default=None     |
| --r_height | Image height resize.                   default=None     |
| --spatial  | Add spatial information to the algorithm. default=False |
| --blur     | Blur the output image.                 default=None     |
