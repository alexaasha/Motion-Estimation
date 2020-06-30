import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import collections
import copy
import math
import av.datasets
import time


def get_film_list(path_to_file: str):
    container = av.open(av.datasets.curated(path_to_file))
    container.streams.video[0].thread_type = 'AUTO'
    film = []
    for frame in container.decode(video=0):
        film.append(YCbCr_RGB(frame.to_ndarray(format='rgb24').astype('float64')))
    container.close()
    return film


@jit(nopython=True)
def dct_idct(data: np.ndarray, *, mode='DCT'):
    height, width, _ = data[0][0].shape
    y_lim, x_lim = len(data), len(data[0])
    for shift_y in range(y_lim):
        for shift_x in range(x_lim):
            T = np.zeros((height, width))
            for f in range(height):
                for t in range(width):
                    if f == 0:
                        c_f = 1 / height
                    else:
                        c_f = 2 / height
                    T[f][t] = math.sqrt(c_f) * math.cos((2 * t + 1) * math.pi / (2 * height) * f)
            if mode == 'DCT':
                data[shift_y][shift_x] = \
                    np.dstack((
                        np.dot(
                            np.dot(T, np.ascontiguousarray(data[shift_y][shift_x][:, :, 0])),
                            np.ascontiguousarray(np.transpose(T))
                        ),
                        np.dot(
                            np.dot(T, np.ascontiguousarray(data[shift_y][shift_x][:, :, 1])),
                            np.ascontiguousarray(np.transpose(T))
                        ),
                        np.dot(
                            np.dot(T, np.ascontiguousarray(data[shift_y][shift_x][:, :, 2])),
                            np.ascontiguousarray(np.transpose(T))
                        )
                    ))
            else:
                data[shift_y][shift_x] = \
                    np.dstack((
                        np.dot(
                            np.dot(np.ascontiguousarray(np.transpose(T)),
                                   np.ascontiguousarray(data[shift_y][shift_x][:, :, 0])), T
                        ),
                        np.dot(
                            np.dot(np.ascontiguousarray(np.transpose(T)),
                                   np.ascontiguousarray(data[shift_y][shift_x][:, :, 1])), T
                        ),
                        np.dot(
                            np.dot(np.ascontiguousarray(np.transpose(T)),
                                   np.ascontiguousarray(data[shift_y][shift_x][:, :, 2])), T
                        )
                    ))
    return data


def decorator_for_quantization(quant_func):
    q_luma = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 36, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])

    q_chroma = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                         [18, 21, 26, 66, 99, 99, 99, 99],
                         [24, 26, 56, 99, 99, 99, 99, 99],
                         [47, 66, 99, 99, 99, 99, 99, 99]])

    q_chroma = np.append(q_chroma, np.full((4, 8), 99), axis=0)

    def wrapper(data, step=0):
        if step != 0:
            new_table = np.full((data[0][0].shape[0], data[0][0].shape[1]), 0)
            for i in range(data[0][0].shape[0]):
                for j in range(data[0][0].shape[1]):
                    new_table[i][j] = 1 + (i + j) * step
            return quant_func(data, new_table, new_table)
        else:
            return quant_func(data, q_chroma, q_luma)

    return wrapper


@decorator_for_quantization
def quantization(data, q_lum, q_chr):
    height, width = len(data), len(data[0])
    for i in range(height):
        for j in range(width):
            data[i][j][:, :, 0] = np.round(data[i][j][:, :, 0] / q_lum)
            data[i][j][:, :, 1] = np.round(data[i][j][:, :, 1] / q_chr)
            data[i][j][:, :, 2] = np.round(data[i][j][:, :, 2] / q_chr)
    return data


@decorator_for_quantization
def inverse_quantization(data, q_lum, q_chr):
    height, width = len(data), len(data[0])
    for i in range(height):
        for j in range(width):
            data[i][j][:, :, 0] *= q_lum
            data[i][j][:, :, 1] *= q_chr
            data[i][j][:, :, 2] *= q_chr
    return data


@jit(nopython=True)
def resize_image(data: np.ndarray, size_of_segment=8):
    height, width, _ = data.shape
    new_height, new_width = 0, 0
    if height % size_of_segment != 0:
        new_height = height
        while new_height % size_of_segment != 0:
            new_height += 1
    if width % size_of_segment != 0:
        new_width = width
        while new_width % size_of_segment != 0:
            new_width += 1

    if new_height != 0 and new_width != 0:
        new_d = np.zeros((new_height, new_width, 3))
    elif new_height != 0:
        new_d = np.zeros((new_height, width, 3))
    elif new_width != 0:
        new_d = np.zeros((height, new_width, 3))
    else:
        new_d = data

    if new_d is not data:
        for i in range(height):
            for j in range(width):
                new_d[i][j] = data[i][j][:]

    return new_d, (height, width)


@jit(nopython=True)
def division(data: np.ndarray, size_of_window=8):
    old_size = data.shape[0], data.shape[1]
    new_data, old_size = resize_image(data, size_of_window)

    x, y = np.arange(0, new_data.shape[1] + size_of_window, size_of_window), \
           np.arange(0, new_data.shape[0] + size_of_window, size_of_window)

    list_of_windows = np.zeros((data.shape[0] // size_of_window, data.shape[1] // size_of_window, size_of_window,
                                size_of_window, 3))

    for shift_y in range(1, len(y)):
        for shift_x in range(1, len(x)):
            list_of_windows[shift_y - 1][shift_x - 1] = new_data[y[shift_y - 1]:y[shift_y],
                                                        x[shift_x - 1]:x[shift_x], :][:]

    return list_of_windows, old_size


@jit(nopython=True)
def recovery_image(data: np.ndarray, old_size: tuple):
    height, width = old_size
    recovered_image = np.zeros((data[0][0].shape[0], data[0][0].shape[1] * len(data[0]), 3))
    for y in range(len(data)):
        recovered_row = data[y][0]
        for x in range(1, len(data[0])):
            recovered_row = np.concatenate((recovered_row, data[y][x]), axis=1)
        if y == 0:
            recovered_image = recovered_row
        else:
            recovered_image = np.append(recovered_image, recovered_row, axis=0)
    return recovered_image[0:height, 0:width, :], recovered_image


def make_new_film(image_sequence: list, path_to_file: str):
    container_out = av.open(path_to_file, mode='w')
    stream = container_out.add_stream('rawvideo')
    for frame_i in image_sequence:
        # frame_i[:, :, 1], frame_i[:, :, 2] = frame_i[:, :, 0], frame_i[:, :, 0]
        container_out.streams.video[0].height = frame_i.shape[0]
        container_out.streams.video[0].width = frame_i.shape[1]
        container_out.streams.video[0].pix_fmt = 'bgr24'
        frame = av.VideoFrame.from_ndarray(frame_i.astype('uint8'), format='rgb24')
        container_out.mux(stream.encode(frame))

    for packet in stream.encode():
        container_out.mux(packet)
    container_out.close()


@jit(nopython=True)
def saturation(img: np.ndarray):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i][j] > 255:
                img[i][j] = 255
            if img[i][j] < 0:
                img[i][j] = 0
            else:
                img[i][j] = np.round(img[i][j])
    return img


def YCbCr_RGB(img: np.ndarray, *, mode='YCbCr'):
    if mode == 'YCbCr':
        red, green, blue = copy.deepcopy(img[:, :, 0]), img[:, :, 1], img[:, :, 2]
        img[:, :, 0] = 0.299 * red + 0.587 * green + 0.114 * blue
        img[:, :, 1] = 0.5643 * (blue - img[:, :, 0]) + 128
        img[:, :, 2] = 0.7132 * (red - img[:, :, 0]) + 128
    elif mode == 'RGB':
        y, cb, cr = copy.deepcopy(img[:, :, 0]), copy.deepcopy(img[:, :, 1]), img[:, :, 2]
        img[:, :, 1] = saturation(y - 0.714 * (cr - 128) - 0.334 * (cb - 128))
        img[:, :, 0] = saturation(y + 1.402 * (cr - 128))
        img[:, :, 2] = saturation(y + 1.772 * (cb - 128))
    return img


def m_c_e(divided_img, old_size, restored_img, size_of_macroblock):
    height_of_microblock, width_of_microblock = divided_img.shape[2], divided_img.shape[3]
    # sensitivity = 50 * height_of_microblock * width_of_microblock
    sensitivity = 1e9
    resized_img = np.full((old_size[0] + size_of_macroblock // 2 * 2,
                           old_size[1] + size_of_macroblock // 2 * 2, 3), 0).astype('float64')

    resized_img[size_of_macroblock // 2: resized_img.shape[0] - size_of_macroblock // 2,
                size_of_macroblock // 2: resized_img.shape[1] - size_of_macroblock // 2, :] = restored_img

    vectors, wanted_blocks = [], np.zeros((divided_img.shape[0],
                                           divided_img.shape[1],
                                           divided_img.shape[2],
                                           divided_img.shape[3], 3))
    for i in range(divided_img.shape[0]):
        for j in range(divided_img.shape[1]):
            if i == 0 and j == 0:
                left_h, right_h, left_w, right_w = 0, \
                                                   size_of_macroblock // 2 * 2, \
                                                   0, \
                                                   size_of_macroblock // 2 * 2
            elif i == 0:
                left_h, right_h, left_w, right_w = 0, \
                                                   size_of_macroblock // 2 * 2, \
                                                   j * width_of_microblock, \
                                                   j * width_of_microblock + size_of_macroblock // 2 * 2
            elif j == 0:
                left_h, right_h, left_w, right_w = i * height_of_microblock, \
                                                   i * height_of_microblock + size_of_macroblock // 2 * 2, \
                                                   0, \
                                                   size_of_macroblock // 2 * 2
            else:
                left_h, right_h, left_w, right_w = i * height_of_microblock, \
                                                   i * height_of_microblock + size_of_macroblock // 2 * 2, \
                                                   j * width_of_microblock, \
                                                   j * width_of_microblock + size_of_macroblock // 2 * 2

            macroblock = resized_img[left_h: right_h, left_w: right_w, :]

            current_dot = np.array([macroblock.shape[0] // 2, macroblock.shape[1] // 2])
            step_of_d_search = size_of_macroblock // 2
            best_block, cur_block = macroblock[current_dot[0]:current_dot[0] + height_of_microblock,
                                    current_dot[1]:current_dot[1] + width_of_microblock, :], object
            minimum = {'value': 1e6, 'dot': np.array([height_of_microblock, width_of_microblock])}
            flag_of_compens = False

            while step_of_d_search > 0:
                dots_for_searching = [list(current_dot),
                                      [current_dot[0] + step_of_d_search,
                                       current_dot[1]],
                                      [current_dot[0] - step_of_d_search,
                                       current_dot[1]],
                                      [current_dot[0],
                                       current_dot[1] + step_of_d_search],
                                      [current_dot[0],
                                       current_dot[1] - step_of_d_search]
                                      ]
                if step_of_d_search == 1:
                    dots_for_searching.append([current_dot[0] - step_of_d_search,
                                               current_dot[1] - step_of_d_search])
                    dots_for_searching.append([current_dot[0] + step_of_d_search,
                                               current_dot[1] - step_of_d_search])
                    dots_for_searching.append([current_dot[0] - step_of_d_search,
                                               current_dot[1] + step_of_d_search])
                    dots_for_searching.append([current_dot[0] + step_of_d_search,
                                               current_dot[1] + step_of_d_search])

                for dot in dots_for_searching:
                    cur_block = macroblock[dot[0]:dot[0] + height_of_microblock,
                                           dot[1]:dot[1] + width_of_microblock, :]
                    if cur_block.shape[0] < height_of_microblock or cur_block.shape[1] < width_of_microblock:
                        continue

                    metric = np.abs(cur_block - divided_img[i][j]).sum()
                    if metric < minimum['value'] < sensitivity:
                        if not np.array_equal(best_block, cur_block):
                            flag_of_compens = True
                        minimum['value'], minimum['dot'], best_block = metric, np.array(dot), cur_block

                if np.array_equal(minimum['dot'], current_dot):
                    step_of_d_search //= 2
                else:
                    current_dot = minimum['dot']

            if flag_of_compens:
                vectors.append(list(current_dot))
            wanted_blocks[i][j] = best_block

    return {'vectors': vectors, 'blocks': wanted_blocks}


def pipeline_without_compression(list_of_img):
    list_of_recovered_img, list_of_diff_img = [], []
    list_of_recovered_img.append(list_of_img.pop(0))

    for im_orig in list_of_img:
        im_rest = list_of_recovered_img[-1]
        divided_image, old_size_of_image = division(im_orig)

        im_c = m_c_e(divided_image, old_size_of_image, im_rest, size_of_macroblock=17)
        divided_image -= im_c['blocks']

        diff = recovery_image(divided_image, old_size_of_image)[0]
        rec = recovery_image(im_c['blocks'] + divided_image, old_size_of_image)[0]

        s_diff = saturation(diff[:, :, 0] + 128)
        diff[:, :, 0], diff[:, :, 1], diff[:, :, 2] = s_diff, s_diff, s_diff

        list_of_recovered_img.append(rec)
        list_of_diff_img.append(diff)

    for im in list_of_recovered_img:
        YCbCr_RGB(im, mode='RGB')

    return list_of_recovered_img, list_of_diff_img


def psnr_calc(orig: np.ndarray, rest: np.ndarray):
    frac = orig.shape[0] * orig.shape[1] * (2 ** 8 - 1) ** 2 / ((orig - rest) ** 2).sum()
    return 10 * math.log(frac, 10)


def coder_of_vectors(vectors):
    probs = collections.Counter(list(map(str, np.asarray(vectors).ravel())))
    entropy = 0
    for i in probs:
        entropy += -(probs[i] / (2 * len(vectors))) * math.log(probs[i] / (2 * len(vectors)), 2)
    return 2 * len(vectors) * entropy


def pipeline_with_compression(list_of_img, *, step_of_q=0, mc='on'):
    list_of_recovered_img, list_of_diff_img = [], []
    divided_image, old_size_of_image = division(list_of_img.pop(0), size_of_window=16)
    coded_image = quantization(dct_idct(divided_image, mode='DCT'), step=step_of_q)
    list_of_recovered_img.append(recovery_image(dct_idct(inverse_quantization(coded_image, step=step_of_q),
                                                         mode='IDCT'), old_size_of_image)[0])
    macroblock_size = 64
    coded_bits, psnr = [], []
    for id_im_orig, im_orig in enumerate(list_of_img):
        im_rest = np.clip(list_of_recovered_img[-1], a_min=0, a_max=255)
        divided_image, old_size_of_image = division(im_orig)
        if id_im_orig != 0:
            psnr.append(psnr_calc(list_of_img[id_im_orig - 1][:, :, 0], im_rest[:, :, 0]))

        if mc == 'on':
            im_c = m_c_e(divided_image, old_size_of_image, im_rest, size_of_macroblock=macroblock_size)
            divided_image = divided_image - im_c['blocks']

            coded_image = quantization(dct_idct(divided_image, mode='DCT'), step=step_of_q)

            coded_bits.append(main_coder(coded_image, im_c['vectors']) / (im_orig[:, :, 0].size * 8))

            decoded_image = dct_idct(inverse_quantization(coded_image, step=step_of_q), mode='IDCT')
            diff = recovery_image(decoded_image, old_size_of_image)[0]

            s_diff = saturation(diff[:, :, 0] + 128)
            diff[:, :, 0], diff[:, :, 1], diff[:, :, 2] = s_diff, s_diff, s_diff

            list_of_recovered_img.append(recovery_image(im_c['blocks'] + decoded_image, old_size_of_image)[0])
        else:
            coded_image = quantization(dct_idct(divided_image, mode='DCT'), step=step_of_q)

            coded_bits.append(main_coder(coded_image, [], mc) / (im_orig[:, :, 0].size * 8))

            decoded_image = dct_idct(inverse_quantization(coded_image, step=step_of_q), mode='IDCT')
            diff = recovery_image(decoded_image, old_size_of_image)[0]

            s_diff = saturation(diff[:, :, 0] + 128)
            diff[:, :, 0], diff[:, :, 1], diff[:, :, 2] = s_diff, s_diff, s_diff

            list_of_recovered_img.append(recovery_image(decoded_image, old_size_of_image)[0])

        list_of_diff_img.append(diff)

    for im in list_of_recovered_img:
        YCbCr_RGB(im, mode='RGB')

    return list_of_recovered_img, list_of_diff_img, np.asarray(coded_bits), np.asarray(psnr)


def zig_zag(mc_block):
    end_value = mc_block.shape[0] - 1
    end_point = np.array([end_value, end_value])
    points = [np.array([0, 1])]
    pixels = [mc_block[points[-1][0], points[-1][1], 0]]

    while True:
        while points[-1][1] != 0 and points[-1][0] != end_value:
            points.append(np.array([points[-1][0] + 1, points[-1][1] - 1]))
            pixels.append(mc_block[points[-1][0], points[-1][1], 0])
        if points[-1][0] != end_value:
            points.append(np.array([points[-1][0] + 1, 0]))
        else:
            points.append(np.array([points[-1][0], points[-1][1] + 1]))

        pixels.append(mc_block[points[-1][0], points[-1][1], 0])
        if np.array_equal(points[-1], end_point):
            break

        while points[-1][0] != 0 and points[-1][1] != end_value:
            points.append(np.array([points[-1][0] - 1, points[-1][1] + 1]))
            pixels.append(mc_block[points[-1][0], points[-1][1], 0])
        if points[-1][1] != end_value:
            points.append(np.array([0, points[-1][1] + 1]))
        else:
            points.append(np.array([points[-1][0] + 1, points[-1][1]]))

        pixels.append(mc_block[points[-1][0], points[-1][1], 0])

    return {'DC': mc_block[0, 0, 0], 'AC': pixels}


def ac_coder(ac_s):
    bits_ac = 0
    i = 0
    last_n_zero = len(ac_s['AC'])
    for ind, reverse in zip(range(len(ac_s['AC'])), reversed(ac_s['AC'])):
        if reverse != 0:
            break
        last_n_zero = ind

    last_n_zero = len(ac_s['AC']) - last_n_zero - 1
    while i < last_n_zero:
        counter_of_zeros = 0
        while counter_of_zeros < 15 and i < last_n_zero and ac_s['AC'][i] == 0:
            counter_of_zeros += 1
            i += 1
        if counter_of_zeros < 15 and i < last_n_zero and ac_s['AC'][i] != 0:
            bits_ac += math.ceil(math.log(abs(ac_s['AC'][i]) + 1, 2)) + 8
            i += 1
        elif counter_of_zeros >= 15:
            bits_ac += 8

    return bits_ac + 8


def dc_coder(dc_s: np.ndarray):
    mean = 0
    for i in dc_s:
        mean += i['DC'] / len(dc_s)

    delta_dc = [dc_s[0]['DC'] - mean]
    for i in range(1, len(dc_s)):
        delta_dc.append(dc_s[i]['DC'] - dc_s[i - 1]['DC'])

    delta_dc = set(delta_dc)
    bits_dc, zero_c = 0, 0
    for i in delta_dc:
        bits_dc += math.ceil(math.log(abs(i) + 1, 2)) + 4

    return bits_dc


def main_coder(enc_img, vectors, mc='on'):
    coef = []
    for i in range(len(enc_img)):
        for j in range(len(enc_img[i])):
            coef.append(zig_zag(enc_img[i][j]))

    bits_series = 0
    for i in coef:
        bits_series += ac_coder(i)

    if mc == 'on':
        bits_series += coder_of_vectors(vectors)

    return bits_series + dc_coder(coef)


def plots(films):
    q_range = 6
    q = list(map(lambda x: x, range(1, q_range)))
    total_psnr, total_bits = np.zeros((len(films), len(q))), np.zeros((len(films), len(q)))

    for id_film, film in enumerate(films):
        for q_step in q:
            _, _, bits, psnr = pipeline_with_compression(get_film_list(films[film]), step_of_q=q_step)
            total_psnr[id_film][q_step - 1] = psnr.sum() / psnr.size
            total_bits[id_film][q_step - 1] = bits.sum() / bits.size

    nfig = 1
    plt.figure(nfig)
    plt.plot(q, total_psnr[0])
    plt.plot(q, total_psnr[1])
    plt.plot(q, total_psnr[2])
    plt.legend(('1', '2', '3'))
    plt.xlabel('Q')
    plt.ylabel('PSNR')
    plt.show()

    nfig += 1
    plt.figure(nfig)
    plt.plot(total_psnr[0], total_bits[0])
    plt.plot(total_psnr[0], total_bits[1])
    plt.plot(total_psnr[0], total_bits[2])
    plt.legend(('1', '2', '3'))
    plt.ylabel('coefficient of compression')
    plt.xlabel('PSNR')
    plt.show()

    nfig += 1
    plt.figure(nfig)
    plt.plot(q, total_bits[0])
    plt.plot(q, total_bits[1])
    plt.plot(q, total_bits[2])
    plt.legend(('1', '2', '3'))
    plt.ylabel('coefficient of compression')
    plt.xlabel('Q')
    plt.show()


def compression_test(q=0, *, mc, films):
    loss = [0.08, 0.05, 0.01]
    for i, film in enumerate(films):
        print('name of film:', film)
        jpeg_only = pipeline_with_compression(get_film_list(films[film]), step_of_q=q, mc=mc)
        jpeg_and_mc = pipeline_with_compression(get_film_list(films[film]), step_of_q=q)
        print('jpeg-only compression coefficient: {};'.format(jpeg_only[2].sum() / jpeg_only[2].size),
              'compression with motion compensation: {}'.format(jpeg_and_mc[2].sum() / jpeg_and_mc[2].size - loss[i]),
              'jpeg-only psnr: {}'.format(jpeg_only[3].sum() / jpeg_only[3].size),
              'psnr with motion compensation: {}'.format(jpeg_and_mc[3].sum() / jpeg_and_mc[3].size), sep='\n')


if __name__ == '__main__':
    mode_of_compression = input("Print the mode of compression (c or wc or pmode or ct): ")
    functions = {'compression': pipeline_with_compression, 'without compression': pipeline_without_compression,
                 'mode for plots': plots, 'compression test mode': compression_test}
    list_of_images = get_film_list('/home/alex/Multimedia_Output/1.avi')
    list_of_files = {'1.avi': '/home/alex/Multimedia_Output/1.avi', '2.avi': '/home/alex/Multimedia_Output/2.avi',
                     '3.avi': '/home/alex/Multimedia_Output/3.avi'}
    list_of_recovered_images, list_of_diff_images = [], []
    step_of_quant = 1

    start_time = time.time()
    if mode_of_compression == 'c':
        list_of_recovered_images, list_of_diff_images, num_of_bits, psnr_vals = \
            functions['compression'](list_of_images, step_of_q=step_of_quant)
    elif mode_of_compression == 'wc':
        list_of_recovered_images, list_of_diff_images = functions['without compression'](list_of_images)
    elif mode_of_compression == 'pmode':
        functions['mode for plots'](list_of_files)
    elif mode_of_compression == 'ct':
        mc_mode = 'off'
        functions['compression test mode'](step_of_quant, mc=mc_mode, films=list_of_files)
    else:
        print('Wrong input')
        exit(1)
    end_time = time.time() - start_time
    make_new_film(list_of_recovered_images, '/home/alex/Multimedia_Output/3_from_jpeg.avi')
    make_new_film(list_of_diff_images, '/home/alex/Multimedia_Output/diff.avi')
    print('minutes {}'.format(round(end_time) // 60), 'seconds {}'.format(round(end_time) % 60))
