import os
import glob
import math
import cv2 as cv
import numpy as np
from numba import njit


def make_a_pyramid(image1: np.ndarray, image2: np.ndarray, height: int):
    """
    A function for constructing Gaussian pyramid

    :param image1: current image
    :param image2: base image
    :param height: height of pyramid
    :return: pyramids for current and base images
    """

    pyramid1, pyramid2 = [image1], [image2]

    for i in range(height):
        g1 = cv.GaussianBlur(pyramid1[i], (5, 5), 1)
        g2 = cv.GaussianBlur(pyramid2[i], (5, 5), 1)

        pyramid1.append(g1[::2, ::2])
        pyramid2.append(g2[::2, ::2])

    return pyramid1, pyramid2


def write_flo(u, v, path_to_file, filename):
    """
    A function for writing vectors field to a .flo file

    :param u: projection of a vector on the x axis
    :param v: projection of a vector on the y axis
    :param path_to_file: path to .flo file
    :param filename: filename (.flo)
    """

    height, width = u.shape

    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)

    path_to_file += filename
    with open(path_to_file, 'wb') as f:
        f.write('PIEH'.encode('ascii'))
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        tmp = np.zeros((height, width * 2))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)


def color(flo_file_dir):
    """
    A function for drawing vectors field

    :param flo_file_dir: path to flo files
    """

    for root, _, _ in os.walk(flo_file_dir):
        for flo in glob.glob(root + "/*.flo"):
            add_inf = flo.split('/')
            store_path = '/home/alex/multimedia_output/flow_color/' + add_inf[-2] + '/'
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            os.system('~/multimedia_output/flow_code/C/color_flow ' + flo + ' ' + store_path +
                      add_inf[-1].replace('.flo', '.png'))
            print(flo.replace('.flo', '.png'))


@njit
def compare_frames(im1, im2):
    """
    A function for comparing frames

    :param im1: current image
    :param im2: base image
    :return: result of comparison
    """

    return ((im1 - im2) ** 2).sum() / im1.size


def warp_bilinear(im, u, v):
    """
    A function for warping image

    :param im: current image
    :param u: projection of a vector on the x axis
    :param v: projection of a vector on the y axis
    :return: warped image
    """

    old_size = im.shape
    new_size = im.shape[1] * 2, im.shape[0] * 2
    im = cv.resize(im, dsize=new_size)
    x_grid, y_grid = np.meshgrid(np.arange(0, im.shape[1], 2), np.arange(0, im.shape[0], 2))

    return im[np.round(v + y_grid).flatten().astype(int), np.round(u + x_grid).flatten().astype(int)].reshape(old_size)


def clipping(vect_field, gu, gv):
    """
    A function for clipping vectors field

    :param vect_field: vectors field
    :param gu: projection of a vector on the x axis
    :param gv: projection of a vector on the y axis
    :return: clipped field
    """

    x_grid, y_grid = np.meshgrid(np.arange(vect_field[1]), np.arange(vect_field[0]))
    c_u, c_v = np.clip(gu + x_grid, 0, vect_field[1] - 1), \
               np.clip(gv + y_grid, 0, vect_field[0] - 1)

    gu = c_u - x_grid
    gv = c_v - y_grid

    return gu, gv


@njit
def upsample(vectors, dsize):
    """
    A function for upsampling

    :param vectors: vectors field
    :param dsize: size of new field
    :return: upsampled field
    """

    new_field = np.zeros((vectors.shape[0] * 2, vectors.shape[1] * 2))

    new_field[::2, ::2] = vectors
    new_field[::2, 1::2] = vectors
    new_field[1::2, :] = new_field[::2, :]

    return new_field[:dsize[1], :dsize[0]]


@njit
def recovery(arr, old_size):
    """
    A function for recovery image or vectors field

    :param arr: current image or vectors field
    :param old_size: old size
    :return: recovered object
    """

    y_dim, x_dim = old_size
    return arr[:y_dim, :x_dim]


@njit
def resize(arr, size_of_block):
    """
    A function for resize image or vectors field

    :param arr: object to resize
    :param size_of_block: size of block
    :return: resized object
    """

    height, width = arr.shape
    new_height, new_width = height, width

    while new_height % size_of_block[0] != 0:
        new_height += 1

    while new_width % size_of_block[1] != 0:
        new_width += 1

    new_arr = np.zeros((new_height, new_width))
    new_arr[:height, :width] = arr

    return new_arr, (height, width)


def psnr(orig: np.ndarray, rest: np.ndarray):
    """
    A function for PSNR calculation

    :param orig: original image
    :param rest: restored image
    :return: result of comparison
    """

    frac = orig.shape[0] * orig.shape[1] * (2 ** 8 - 1) ** 2 / ((orig - rest) ** 2).sum()
    return 10 * math.log(frac, 10)


def read_flo(file):
    """
    A function for reading vectors from .flo file

    :param file: file
    :return: vectors field
    """

    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == 202021.25, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()

    return flow


@njit
def flow_comparator(my_flow, orig_flow):
    """
    A function for vectors field comparison

    :param my_flow: computed vectors
    :param orig_flow: original vectors
    :return: result of comparison
    """

    epe = np.sqrt((my_flow[..., 0] - orig_flow[..., 0]) ** 2 +
                  (my_flow[..., 1] - orig_flow[..., 1]) ** 2).sum()
    return epe / my_flow[..., 0].size
