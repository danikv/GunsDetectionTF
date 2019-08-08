import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

images = []
id = 0
for filename in glob('guns/*.jpg'):
    img1 = cv2.imread(filename)          # queryImage
    # img2 = cv2.imread('gun_fr350_obj0.jpg') # trainImage
    # Initiate ORB detector

    name = 'guns_{}'.format(id)
    if not images:
        images.append(img1)
        cv2.imwrite('guns_filtered/{}.jpg'.format(name), img1)
        id += 1
        continue

    shouldUpdate = True
    img1 = cv2.resize(img1, (300, 300), interpolation=cv2.INTER_LINEAR)
    for img2 in images:
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        if des1 is None:
            shouldUpdate = False
            continue

        kp2, des2 = orb.detectAndCompute(img2,None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        # print(type(des1))
        # print(type(des2))
        if des1 is None:
            shouldUpdate = False
            continue

            # cv2.imshow('a', img1)
            # cv2.waitKey()
        # if des2 is None:
        #     cv2.imshow('b', img2)
        #     cv2.waitKey()
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        if not matches:
            continue

        filtered = [x.distance for x in matches if x.distance < 75]
        if not filtered:
            continue

        avg = sum(filtered) / len(filtered)
        # maxmin = max(map(lambda x: x.distance, matches))
        if avg < 61:
            shouldUpdate = False
            break

    if shouldUpdate == True:
        images.append(img1)
        cv2.imwrite('guns_filtered/{}.jpg'.format(name), img1)
        id += 1


    # print(avg)
# Draw first 10 matches.
# img4 = []
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20], flags=2, outImg=None)
# plt.imshow(img3),plt.show()
