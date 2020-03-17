# import the necessary packages
from scipy.spatial import distance as dist
import cv2
import numpy as np
import imutils
import os
import math
from imutils import perspective

proc_rate = 0

path = '1/'
criteria_picture = '1/' + '67800.jpg'
res_path = 'res-img/'
static_path = 'static/ImageRetrieval/assets/'

min_rate = 1.16
NResult = 10
canny_thres = (30, 60)


def global_param_setting(mpath, mcriteria_picture, mres_path, mmin_rate, mNResult):
    global path, criteria_picture, res_path, min_rate, NResult
    print('---global param setting called---')
    path = mpath
    criteria_picture = mcriteria_picture
    res_path = mres_path
    min_rate = mmin_rate
    NResult = mNResult


def remove_outrect(gray):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_border = 1
    (h, w) = gray.shape

    gray_crop = gray[out_border:h - out_border, out_border:w - out_border]
    return gray_crop.copy()


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def get_binary(gray):
    thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    """cv2.imshow('thres', thres)
    cv2.waitKey(0)
    outline = np.zeros(gray.shape, dtype='uint8')
    cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(outline, [cnts], -1, 255, -1)"""
    return thres

def area_diff(bin1, bin2):
    """(h1, w1) = bin1.shape
    (h2, w2) = bin2.shape
    if (h1 != h2 or w1 != w2):
        print ('something wrong')
        assert (0)
    b1 = (bin1 > 0).astype(np.uint8)
    b2 = (bin2 > 0).astype(np.uint8)
    inter = (b1 == b2).astype(np.uint8)
    inter = np.where((b1 > 0), inter, 0)

    inter_area = inter.sum()
    b1_area = b1.sum()
    b2_area = b2.sum()
    score = (1.0 - 2 * inter_area / (b1_area + b2_area))"""
    diff = (bin1 != bin2).astype(np.uint8)
    return diff.sum()

def transform_img(img, box, dst_cores = np.float32([[0.0, 0.0], [0.0, 120.0], [160.0, 120.0]])):

    pt0 = box[0]
    pt1 = box[1]
    pt2 = box[2]
    pt3 = box[3]
    dis1 = math.sqrt((pt0[0] - pt1[0]) * (pt0[0] - pt1[0]) + (pt0[1] - pt1[1]) * (pt0[1] - pt1[1]))
    dis2 = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    if dis1 == 0 or dis2 == 0:
        rate = 100000
    else:
        rate = dis1 / dis2
        if rate < 1.0:
            rate = 1.0 / rate

    src0 = pt0
    src1 = pt1
    src2 = pt2
    if dis1 > dis2:
        src0 = pt1
        src1 = pt2
        src2 = pt3

    src_cores = np.float32([[src0[0],src0[1]], [src1[0],src1[1]], [src2[0],src2[1]]])
    warp_mat = cv2.getAffineTransform(src_cores, dst_cores)
    ww = (int)(dst_cores[2][0] - dst_cores[0][0])
    hh = (int)(dst_cores[1][1] - dst_cores[0][1])
    #dst_img = np.zeros((hh, ww), dtype=np.uint8)
    dst_img = cv2.warpAffine(img, warp_mat, (ww, hh))

    # fliping image to y    ---------------------------------------
    hh2 = (int)(hh/2)
    ww2 = (int)(ww/2)
    limg = dst_img[0:hh, 0:ww2]
    rimg = dst_img[0:hh, ww2:ww]
    larea = limg.sum()
    rarea = rimg.sum()
    if rarea < larea:
        dst_img = cv2.flip(dst_img, 1)
        #cv2.imshow ('yaxis', dst_img)
        #cv2.waitKey(0)

    # fliping image to x -------------------------------------------
    uimg = dst_img[0:hh2, 0:ww]
    dimg = dst_img[hh2:hh, 0:ww]
    uarea = uimg.sum()
    darea = dimg.sum()
    if darea < uarea:
        dst_img = cv2.flip(dst_img, 0)
        #cv2.imshow('xaxis', dst_img)
        #cv2.waitKey(0)

    return dst_img, rate

def get_edges_canny(gray, bScaned = False, bDraw = False):
    gray = remove_outrect(gray)
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)

    # perform edge detection, then close the gaps of edges
    # by dilation + erosion
    edge = cv2.Canny(smoothed, canny_thres[0], canny_thres[1])
    edge = cv2.dilate(edge, None, iterations=3)
    edge = cv2.erode(edge, None, iterations=3)

    #if bDraw:
    #    cv2.imshow('edge', edge)
    #    cv2.waitKey(0)

    outline = np.zeros(edge.shape, dtype='uint8')
    cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(outline, [cnt], -1, 255, -1)

    if bScaned:
        print ('----------------------------------------------------------------------------')
        outline = (outline > 0).astype(np.uint8)
        outline = cv2.medianBlur(outline, 9)
        outline = np.where((outline > 0), 255, 0).astype(np.uint8)

    if bDraw:
        cv2.imshow('draw', outline)
        cv2.waitKey(0)

    # compute the rotated bounding  box of the contour
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype='int')

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)

    norm_img, rate = transform_img(outline, box)

    if 0:
        # hu-moment calculation
        moment = cv2.moments(norm_img)
        huMoment = cv2.HuMoments(moment)
        # log transform
        for i in range(7):
            huMoment[i] = -1 * np.copysign(1.0, huMoment[i]) * np.log10(abs(huMoment[i]))

    pt0 = box[0]
    pt1 = box[1]
    pt2 = box[2]
    pt3 = box[3]

    # draw the out bound rectangle box
    if bDraw:
        color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        cv2.line(color, (pt0[0], pt0[1]), (pt1[0], pt1[1]), (0, 0, 255))
        cv2.line(color, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255))
        cv2.line(color, (pt2[0], pt2[1]), (pt3[0], pt3[1]), (0, 0, 255))
        cv2.line(color, (pt3[0], pt3[1]), (pt0[0], pt0[1]), (0, 0, 255))
        cv2.imshow('edge', color)
        cv2.waitKey(0)


    (h, w) = norm_img.shape
    n2 = np.zeros((h+4, w+4), np.uint8)
    n2[2:h+2, 2:w+2] = norm_img

    outline = np.zeros(n2.shape, dtype='uint8')
    cnts = cv2.findContours(n2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    #cv2.drawContours(outline, [cnt], -1, 255, -1)
    #cv2.imshow('new_outline', n2)
    #cv2.waitKey(0)

    return cnt, rate, None, norm_img

def delete_files_in_folder(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def calc_dist(hu1, hu2, method):
    if method == 'moment':
        dist = (hu1 - hu2) * (hu1 - hu2)
        # sum  = (dist[0][0] * 10. + dist[1][0] * 3. + dist[2][0] * 3. + dist[3][0] * 2. +
        #        dist[4][0] + dist[5][0] * 2. + dist[6][0])
        sum = (dist[0][0] + dist[1][0] + dist[2][0] + dist[3][0] + dist[4][0] + dist[5][0] + dist[6][0])
        sum = math.sqrt(sum)
        sum /= 70

        return sum
    elif method == 'rate':
        rate = hu1 / hu2
        if rate < 1: rate = 1 / rate
        return rate


def get_filtered(rates, score, names):
    n = len(rates)
    n_score = []
    n_names = []
    n_rates = []
    for i in range(n):
        if rates[i] < min_rate:
            n_score.append(score[i])
            n_names.append(names[i])
            n_rates.append(rates[i])

    return n_rates, n_score, n_names

def get_similar_names(rates, dists, names, n=NResult):
    n = min(n, len(dists))
    print (n)
    dists = np.array(dists)
    res = []
    for i in range(n):
        idx = np.argmin(dists)
        res.append((rates[idx], dists[idx], names[idx], i + 1))
        dists[idx] = 100000000.0
    return res

def search_engine(startidx, search_method=0):
    global proc_rate

    print (NResult)
    print (search_method)
    delete_files_in_folder(res_path)
    delete_files_in_folder(static_path)

    img = cv2.imread(criteria_picture, cv2.IMREAD_GRAYSCALE)
    (h, w) = img.shape
    img = cv2.resize(img, ((int)(w / 2), (int)(h / 2)))

    cont_criteria, rate_criteria, hu_criteria, im_criteria = get_edges_canny(img, True, False)

    files = [name for name in os.listdir(path)]
    dists = []
    rates = []

    NN = len(files)
    nn = 0

    for file in files:
        file_path = path + file
        print ('-- file name : {}'.format(file))
        cimg = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if np.any(cimg) == None:
            continue
        (h, w) = cimg.shape
        h = (int)(h / 2)
        w = (int)(w / 2)
        cimg = cv2.resize(cimg, (w, h))
        cont, ratev, hu_m, im = get_edges_canny(cimg)

        rate_dist = calc_dist(rate_criteria, ratev, 'rate')
        cont_dist = cv2.matchShapes(cont_criteria, cont, 1, 0.0)

        if 0:
            hu_dist = calc_dist(hu_criteria, hu_m, 'moment')
        im_dist =area_diff(im_criteria, im)

        print('{} : {}, {}, {}'.format(file, rate_dist, cont_dist, im_dist))

        if search_method == 0:
            dists.append(cont_dist)
        elif search_method == 1:
            dists.append(im_dist)
        else:
            dists.append(im_dist/3000 + cont_dist)

        nn += 1
        proc_rate = nn * 100 / NN
        print (proc_rate)

        rates.append(rate_dist)

    nrates, ndists, nfiles = get_filtered(rates, dists, files)
    res = get_similar_names(nrates, ndists, nfiles, NResult)
    idx = 0
    for each in res:
        file_path = path + each[2]
        img = cv2.imread(file_path)
        cv2.imwrite(res_path + each[2], img)
        cv2.imwrite('{}{}.jpg'.format(static_path, idx + startidx), img)
        idx += 1
    print(res)
    return len(res)

