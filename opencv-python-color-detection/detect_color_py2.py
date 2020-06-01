# USAGE
# python detect_color.py --image pokemon_games.png

# import the necessary packages
from __future__ import print_function 
import math
import numpy as np
import argparse
import cv2
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def pdist(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
#image = cv2.imread(args["image"])
image = []
#image.append(cv2.imread('/home/qliu/Dropbox/DataReductionEvaluate-2018-restored/figures/mergeCompress/dpot-1x.png'))
name_hat = "/home/cephuser/opencv-python-color-detection/"
name_tail1 = "original"
#name_tail2 = "original"
name_tail2 = "0_finer"
print (name_tail2)
name1 = name_hat+name_tail1+".png"
name2 = name_hat+name_tail2+".png"
image.append(cv2.imread(name1))
image.append(cv2.imread(name2))
#image.append(cv2.imread("test.png"))
#image.append(cv2.imread("xgca_256_stats_preserving.png"))
#image.append(cv2.imread("xgca_8.png"))
#image.append(cv2.imread("dpot-16.png"))
#image.append(cv2.imread("dpot-32.png"))
#image.append(cv2.imread("dpot-64.png"))

#image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c4-zfp.png"))
#image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c4-sz.png"))
#image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c5-zfp.png"))
#image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c5-sz.png"))
#image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c6-zfp.png"))
#image.append(cv2.imread("/home/qliu/Downloads/compressed-images/c6-sz.png"))

# define the list of boundaries
boundaries = [
	([0, 0, 100], [204, 204, 255]), #red 
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

(lower, upper) = boundaries[0]
print (boundaries[0])

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask


# show the images
#plt.imshow(output)
#plt.show()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;
print (params.thresholdStep)
# Filter by Area.
params.filterByArea = 1
params.minArea = 50

# Filter by Circularity
#params.filterByCircularity = True
#params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = 1
params.minConvexity = 0.3
#params.maxConvexity = 1


# Filter by Inertia
params.filterByInertia = 1
params.minInertiaRatio = 0.1
#params.maxInertiaRatio = 1

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)


mask = []
output = []
gray_image = []
keypoints = []
print(len(image))
for i in range(len(image)):
    time1 = time.time()
    mask.append(cv2.inRange(image[i], lower, upper))
    output.append(cv2.bitwise_and(image[i], image[i], mask = mask[i]))
    gray_image.append(cv2.cvtColor(output[i], cv2.COLOR_BGR2GRAY))
    keypoints.append(detector.detect(gray_image[i]))
    time2 = time.time()
    print ('blob detection time', (time2 - time1))
    print ('image %d blob # %d' %(i, len(keypoints[i])))


    total_diameter = 0
    total_blob_area  = 0
    for k in keypoints[i]:
        total_diameter = total_diameter + k.size
        total_blob_area = total_blob_area + 3.14 * math.pow(k.size/2, 2)
    if len(keypoints[i]):
        print ('avg diameter', total_diameter / len(keypoints[i]))
    else:
        print ('avg diameter', 0)
    print ('aggregate blob area', total_blob_area)

    if i > 0:
        overlap = 0
        for k in keypoints[i]:
            for p in keypoints[0]:
                if pdist(k.pt, p.pt) < (k.size + p.size) * 1.0 / 2.0:
                    overlap = overlap + 1.0
                    break
        if len(keypoints[i]):
            print ('overlap ratio', overlap / len(keypoints[i]))
        else:
            print ('overlap ratio', 0)
        
    im_with_keypoints = cv2.drawKeypoints(cv2.cvtColor(image[i], cv2.COLOR_BGR2RGB), keypoints[i], np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(im_with_keypoints)
    plt.axis('off')
    plt.savefig('/home/cephuser/opencv-python-color-detection/blobed/'+name_tail2+'_blobed.pdf', dpi=600,format='pdf')
    #plt.savefig('/media/sf_shared/blobed/test.pdf', dpi = 600, format='pdf')

    #plt.savefig('xgca_256_after_decimation_blobed.pdf', format='pdf')
    #plt.savefig('xgca_256_stats_preserving_blobed.pdf', format='pdf')
    #plt.show()





#for kp in keypoints:
#    print '(x, y)', int(kp.pt[0]), int(kp.pt[1])
#    print 'diameter', int(kp.size)
#    print 'strength', kp.response

#print 'keypoint type', type(keypoints[0])
# Show keypoints


