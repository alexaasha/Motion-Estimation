import os
import utils
import glob
import imageio
import numpy as np
from math import log2
from numba import njit
from utils import make_a_pyramid, write_flo, clipping, warp_bilinear
from matplotlib import pyplot as plt


@njit
def log_search(im1, im2, occlusion, *, u, v, block_size, s):
    """
    Logarithmic search algorithm

    :param im1: current image
    :param im2: base image
    :param occlusion: bitmap of occlusions
    :param u: projection of a vector on the x axis
    :param v: projection of a vector on the y axis
    :param block_size: block size
    :param s: number of steps
    :return: vectors field
    """

    for i in range(0, im1.shape[0], block_size[0]):
        for j in range(0, im1.shape[1], block_size[1]):
            if occlusion[i, j] == 255:
                continue

            source_block = im1[i:i + block_size[0], j:j + block_size[1]]
            rad = s
            center = int(i + v[i, j]), int(j + u[i, j])

            while rad != 0:
                left, right = (center[0], center[1] - rad * block_size[1]), \
                              (center[0], center[1] + rad * block_size[1])
                up, down = (center[0] - rad * block_size[0], center[1]), \
                           (center[0] + rad * block_size[0], center[1])

                up_left, down_left = (center[0] - rad * block_size[0], center[1] - rad * block_size[1]), \
                                     (center[0] + rad * block_size[0], center[1] - rad * block_size[1])
                up_right, down_right = (center[0] - rad * block_size[0], center[1] + rad * block_size[1]), \
                                       (center[0] + rad * block_size[0], center[1] + rad * block_size[1])

                if rad != 1:
                    dots = [center, left, right, up, down]
                else:
                    dots = [center, left, right, up, down, up_left, down_left, up_right, down_right]

                cur_mse = 1e9
                old_center = center
                for block in dots:
                    if 0 <= block[0] < block[0] + block_size[0] <= im1.shape[0] and \
                       0 <= block[1] < block[1] + block_size[1] <= im1.shape[1]:
                        cur_block = im2[block[0]:block[0] + block_size[0],
                                        block[1]:block[1] + block_size[1]]
                    else:
                        cur_block = None

                    if cur_block is not None:
                        mse = ((source_block - cur_block) ** 2).sum() / source_block.size

                        if cur_mse > mse:
                            cur_mse = mse
                            center = block

                if old_center == center:
                    rad //= 2

            u[i:i + block_size[0], j:j + block_size[1]] = center[1] - int(j + u[i, j])
            v[i:i + block_size[0], j:j + block_size[1]] = center[0] - int(i + v[i, j])

    return u, v


def traversing_pyramid(pyramid1, pyramid2, occlusion, *, d, size_of_block):
    """
    A function for traversing Gauss pyramid

    :param pyramid1: pyramid of first image
    :param pyramid2: pyramid of second image
    :param occlusion: bitmap of occlusions
    :param d: radius of search
    :param size_of_block: size of block
    :return: final vectors field
    """

    if d <= 0 or d > min(pyramid1[0].shape[0], pyramid1[0].shape[1]):
        raise ValueError('Invalid value of diameter: {}, where size of image: {}'.format(d, pyramid1[0].shape))

    if size_of_block > pyramid1[0].shape or size_of_block <= (0, 0):
        raise ValueError('Invalid size of block: {}, where size of image: {}'.format(size_of_block, pyramid1[0].shape))

    gu, gv = np.zeros(pyramid1[-1].shape), np.zeros(pyramid1[-1].shape)
    i = len(pyramid1) - 1
    for im1, im2 in zip(reversed(pyramid1), reversed(pyramid2)):
        im1 = utils.warp_bilinear(im1, gu, gv)
        res_im1, old_size = utils.resize(im1, size_of_block)
        res_im2, _ = utils.resize(im2, size_of_block)

        gu, _ = utils.resize(gu, size_of_block)
        gv, _ = utils.resize(gv, size_of_block)

        s = 2 ** (int(log2(d)) - 1)
        gu, gv = log_search(res_im1, res_im2, occlusion, u=gu, v=gv, block_size=size_of_block, s=s)

        gu = utils.recovery(gu, old_size)
        gv = utils.recovery(gv, old_size)

        if i != 0:
            inv_size = (pyramid1[i - 1].shape[1], pyramid1[i - 1].shape[0])
            gu = utils.upsample(gu, dsize=inv_size)
            gv = utils.upsample(gv, dsize=inv_size)

        gu, gv = clipping(pyramid1[i - 1 if i - 1 >= 0 else 0].shape, gu, gv)
        i -= 1

    gu[occlusion == 255] = 0
    gv[occlusion == 255] = 0

    return gu, gv


def main_func(photoes_dir, flo_dir, ocl_dir, dist, size_of_block):
    """
    The Main function

    :param photoes_dir: directory with frames of video
    :param flo_dir: directory with flo-files
    :param ocl_dir: directory with bitmaps of occlusions
    :param dist: radius of search
    :param size_of_block: size of block
    """

    progress_bar = 0
    print('0%')
    for folders in zip(os.walk(photoes_dir), os.walk(ocl_dir)):
        root_ph, _, _ = folders[0]
        root_ocl, _, _ = folders[1]
        photos = sorted([p for p in glob.glob(root_ph + "/*.png")])
        ocl_name = sorted([p for p in glob.glob(root_ocl + "/*.png")])

        for photo_ind in range(len(photos) - 1):
            pic1, pic2 = np.array(imageio.imread(photos[photo_ind]).astype(float)), \
                         np.array(imageio.imread(photos[photo_ind + 1]).astype(float))

            occlusion = np.array(imageio.imread(ocl_name[photo_ind]).astype(float))

            pic1 = np.dot(pic1[..., :3], [0.2989, 0.5870, 0.1140])
            pic2 = np.dot(pic2[..., :3], [0.2989, 0.5870, 0.1140])

            height = 2
            pyr1, pyr2 = make_a_pyramid(pic1, pic2, height)

            try:
                u, v = traversing_pyramid(pyr1, pyr2, occlusion, d=dist, size_of_block=size_of_block)

                pic_w = warp_bilinear(pic1, u, v)
                plt.imshow(pic_w)
                plt.show()
                m1 = utils.compare_frames(pic_w, pic2)
                m2 = utils.compare_frames(pic1, pic2)

                folders = photos[photo_ind].split('/')
                path_flo = flo_dir + folders[-3] + '_log' + '/' + folders[-2] + '/'

                progress_bar += 1
                print(photos[photo_ind].replace('.png', '.flo') + ' {}%'.format(round((progress_bar / 1087) * 100, 4)))

                write_flo(u, v, path_flo, folders[-1].replace('png', 'flo'))

            except ValueError as err:
                print(repr(err))
                exit(1)


if __name__ == '__main__':
    path_to_photo = 'path to photo'
    path_to_ocl = 'path to bitmaps'
    path_to_flo = 'path to flo files'
    size, distance = (4, 4), 10
    height = 4

    main_func(path_to_photo, path_to_flo, path_to_ocl, distance, size)
