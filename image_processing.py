# capture cropped image sets from training run as received from tmrl
# create visualizer
# train encoder/decoder
# use weights and biases to visualize individual nodes receptive fields and dominant patterns
# show loss as well as before/after images
# progressively constrain innermost layer to maximize compression
# feed encoded lower dimensional data into MLP for learning control

# alternatively, extend the frames grabbed to 5, then train a network to predict the 5th frame from the four previous
# then use this with four live frames to feed into driver AI

import numpy as np
import matplotlib.pyplot as plt
import glob


def display_images(filename):
    images = np.load(filename)
    print(images.shape)

    fig, ax = plt.subplots()
    # ax.imshow(images[0], cmap='gray')
    ax.imshow(images)
    plt.show()

    fig = plt.figure()
    rows = 10
    columns = 10
    for i in range(images.shape[0]):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.imshow(images[i], cmap='gray')

    # Display the grid
    plt.show()


def consolidate_and_dedup_images():
    images = None
    for file in glob.glob("img_archive/*.npy"):
        if images is None:
            images = np.load(file)
        else:
            images = np.append(images, np.load(file), axis=0)
    print(images.shape)
    images = np.unique(images, axis=0)
    print(images.shape)
    np.save("img_archive/consolidated.npy", images)

import time
import cv2
# from tmrl.custom.utils.window import WindowInterface

def grab_data_and_img(self):
    window_interface = WindowInterface("Trackmania")

    img = window_interface.screenshot()[:, :, :3]  # BGR ordering
    img = img[:, :, ::-1]  # reversed view for numpy RGB convention
    np.save("img_archive/{}-{}-{}.npy".format(
                img.shape,
                time.strftime("%Y%m%d-%H%M%S"),
                self.img_archive2.sum()),
            img)

    # if resize_to is not None:  # cv2.resize takes dim as (width, height)
    #     img = cv2.resize(img, self.resize_to)
    # if grayscale:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # else:
    #     img = img[:, :, ::-1]  # reversed view for numpy RGB convention


import cv2
import numpy as np
from matplotlib import pyplot as plt

def warp(img):
    rows, cols, ch = img.shape

    # plt.subplot(211)
    # plt.imshow(img)
    # plt.title('Input')

    xc = 480
    yc = 330

    pts1 = np.float32([[xc-20, yc-20],
                       [xc-20, yc+20],
                       [xc+20, yc+20],
                       [xc+20, yc-20]])

    scale = 0.95
    xc = scale*xc-2500/8
    yc = scale*yc-500/8
    offset = 0.465

    pts2 = np.float32([[xc-2, yc-2],
                       [xc-2+offset, yc+2],
                       [xc+2-offset, yc+2],
                       [xc+2, yc-2]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    width = scale*cols-5000/8
    height = scale*rows-1600/8
    return cv2.warpPerspective(img, M, (int(width), int(height)))


def warp2(img):
    rows, cols, ch = img.shape

    # plt.subplot(211)
    # plt.imshow(img)
    # plt.title('Input')

    xc = 480
    yc = 330

    pts1 = np.float32([[xc-20, yc-20],
                       [xc-20, yc+20],
                       [xc+20, yc+20],
                       [xc+20, yc-20]])

    scale = 8
    xc = scale*xc-2500
    yc = scale*yc-500
    offset = 4.65

    pts2 = np.float32([[xc-20, yc-20],
                       [xc-20+offset, yc+20],
                       [xc+20-offset, yc+20],
                       [xc+20, yc-20]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    width = scale*cols-5000
    height = scale*rows-1600
    return cv2.warpPerspective(img, M, (width, height))


def warpTest():
    img = cv2.imread("img_archive/a1.png")
    plt.subplot(411)
    plt.imshow(warp(img))

    img = cv2.imread("img_archive/a2.png")
    plt.subplot(412)
    plt.imshow(warp(img))

    img = cv2.imread("img_archive/a3.png")
    plt.subplot(413)
    plt.imshow(warp(img))

    img = cv2.imread("img_archive/a4.png")
    plt.subplot(414)
    plt.imshow(warp(img))


    plt.show()


    # Displaying the image
    # while (1):
    #
    #     cv2.imshow('image', img)
    #     if cv2.waitKey(20) & 0xFF == 27:
    #         break

    # cv2.destroyAllWindows()



# for file in glob.glob("img_archive/*.npy"):
#     display_images(file)

# display_images('img_archive/64x64-True-20240319-173536-255307.796875.npy')
# display_images('img_archive/64x64-True-20240319-173537-251772.703125.npy')
# display_images('img_archive/64x64-True-20240319-173538-255820.265625.npy')

# consolidate_and_dedup_images()


warpTest()