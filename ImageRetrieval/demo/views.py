from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from django.http import response
from threading import Thread
import cv2
from locale import atoi, atof
#from .engine import global_param_setting, search_engine, proc_rate
import json
from .models import ImageData #ImageRate, ImageCont, ImageArea
import pickle
import time
import datetime
from dateutil.relativedelta import *
import uuid

import cv2
import numpy as np
import imutils
import os
import math
from imutils import perspective

TEST_IMG = 'E:/cur_work/image_retrieval/shapematching/test/1/945910_original.jpg'
TEST_MODE = True
DEBUG_MODE = False


CAMERA_INDEX = 0

cap_img_path = 'static/'
cap_img_img  = 'capture.jpg'
cap_img_path2 = 'E:/cur_work/image_retrieval/shapematching/test/1/'
cap_img_img2  = '431170_original.jpg'
admin_file_path = 'static/sess.file'

res_path = 'res-img/'
min_rate = 1.16
NResult = 10


bCaptured = False

gCount = 0
proc_rate = 0



static_path = 'static/ImageRetrieval/assets/'
no_img_path = 'static/ImageRetrieval/no_img.png'
g_log_path = 'static/log.txt'

g_save_directory = ''
g_capture_directory = ''


canny_thres = (30, 60)
long_edge  = 1


IMG_NOT_EXIST = -101
DIRECTORY_NOT_EXIST = -102
CAPTURE_FAILED = -103
IMG_PROC_ERROR = -104
DATABASE_ERROR = -110
RES_FOLDER_ERROR = -111
SAVE_OK = 0



BACKUPINTERVAL = '#BackupInterval#'
BACKUPTIME = '#BackupTime#'
BACKUPPERIOD = '#BackupPeriod#'
BACKUPDIRECTORY = '#BackupDirectory#'
SAVEDIRECTORY = '#SaveDirectory#'
CAPTUREDIRECTORY = '#CaptureImageDirectory#'

def ParseAdminConfig(fpath):
    BackupInterval = -1 #state 0
    BackupTime_Hour = -1 #state 1
    BackupTime_Minute = -1
    BackupPeriod = -1 #state 2
    SaveDirectory = '' # state 4
    CaptureDirectory = '' # state 5
    BackupDirectory = [] # state 3
    with open(fpath, 'r') as f:
        state = -1
        while True:
            tstr = f.readline()
            if not tstr: break
            if BACKUPINTERVAL in tstr:
                state = 0
            elif BACKUPTIME in tstr:
                state = 1
            elif BACKUPPERIOD in tstr:
                state = 2
            elif BACKUPDIRECTORY in tstr:
                state = 3
            elif SAVEDIRECTORY in tstr:
                state = 4
            elif CAPTUREDIRECTORY in tstr:
                state = 5
            else:
                if state == 0:
                    BackupInterval = atoi(tstr.strip())
                elif state == 1:
                    tstr = tstr.strip()
                    arr = tstr.split('.')
                    if (len (arr) == 1):
                        arr = tstr.split(':')
                    if (len(arr) != 2):
                        print ('--------------------------Admin config file edit error (time format)-----------------')
                        break
                    BackupTime_Hour = atoi(arr[0])
                    BackupTime_Minute = atoi(arr[1])
                elif state == 2:
                    BackupPeriod = atoi(tstr.strip())
                elif state == 3:
                    BackupDirectory.append(tstr.strip())
                elif state == 4:
                    SaveDirectory = tstr.strip()
                elif state == 5:
                    CaptureDirectory = tstr.strip()
                else:
                    print ('-------- Admin config file edit error -------------------')
                    break
    return {'interval' : BackupInterval, 'hour' : BackupTime_Hour, 'minute' : BackupTime_Minute, 'period' : BackupPeriod, 'directory' : BackupDirectory, 'save_directory' : SaveDirectory,
            'capture_directory' : CaptureDirectory}

def global_param_setting(mres_path, mmin_rate, mNResult):
    global res_path, min_rate, NResult
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
    return diff.sum() / 30000

def area_diff_segment(bin1, bin2):
    (h2, w2) = bin1.shape
    UnitArea = w2 / 10 * 2
    delta = (int)(w2/10)

    inv_score = 0
    for i in range(10):
        ws = i * delta
        we = (i+1) * delta
        roi1 = bin1[0:h2, ws:we]
        roi2 = bin2[0:h2, ws:we]
        diff = (roi1 != roi2).astype(np.uint8)
        diff = diff.sum()
        diff /= UnitArea
        inv_score += (diff * diff)

    return inv_score

def transform_img(img, box, bScanned = False):

    global long_edge

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

    small_edge = 120.0
    long_edge = 280.0
    """if bScanned:
        long_edge = (int)(small_edge * rate)
        if rate > 3: long_edge = small_edge * 3"""

    dst_cores = np.float32([[0.0, 0.0], [0.0, small_edge], [long_edge, small_edge]])


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

    if 0:
        if DEBUG_MODE:
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

    norm_img, rate = transform_img(outline, box, bScaned)

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

def save_one_image_feature(img_name, img_path, date = datetime.date.today()):
    img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_GRAYSCALE)
    if np.any(img) == None: return IMG_NOT_EXIST

    (h, w) = img.shape
    img = cv2.resize(img, ((int)(w/2), (int)(h/2)))

    try:
        cont, rate, hu, norm = get_edges_canny(img, False, False)
    except Exception:
        return IMG_PROC_ERROR

    jdata = {'cont' : cont, 'norm' : norm}
    try:
        #c_uuid = uuid.uuid4()
        foo = ImageData()
        foo.name = img_name
        foo.path = img_path
        foo.date = date
        foo.rate = rate
        foo.set_data(jdata)
        foo.save()
        '''
        foo = ImageRate()
        foo.name = img_name
        foo.path = img_path
        foo.date = datetime.date.today()
        foo.rate = rate
        foo.uuid = c_uuid
        foo.save()

        foo = ImageCont()
        foo.uuid = c_uuid
        foo.set_data(cont)
        foo.save()

        foo = ImageArea()
        foo.uuid = c_uuid
        foo.set_data(norm)
        foo.save()'''
    except Exception:
        return DATABASE_ERROR
    return SAVE_OK

def save_img_folders_feature(pathes):
    global proc_rate
    proc_rate = 0

    files = []
    try:
        for fpath in pathes:
            for fname in os.listdir(fpath):
                full_path = os.path.join(fpath, fname)
                file_ctime = os.path.getctime(full_path)
                file_datetime = datetime.datetime.fromtimestamp(file_ctime)  # get file's timestamp
                files.append([fname, fpath, datetime.date(file_datetime.year, file_datetime.month, file_datetime.day)])
    except Exception:
        return DIRECTORY_NOT_EXIST, "Invalid directories"

    NN = len(files)
    nn = 0
    nSucess = 0
    nNotFound = 0
    NotFound = ''
    nImgProcFailed = 0
    ImgProcFailed = ''
    nDatabaseError = 0
    DatabaseError = ''

    for fpair in files:
        fname = fpair[0]
        fpath = fpair[1]
        fdate = fpair[2]
        res_code = save_one_image_feature(fname, fpath, fdate)
        if res_code == SAVE_OK: nSucess += 1
        elif res_code == IMG_PROC_ERROR:
            nImgProcFailed += 1
            ImgProcFailed += fname + ','
        elif res_code == DATABASE_ERROR:
            nDatabaseError += 1
            DatabaseError += fname + ','
        elif res_code == IMG_NOT_EXIST:
            nNotFound += 1
            NotFound += fname + ','

        nn += 1
        proc_rate = nn * 100 / NN

    if len(NotFound) > 0: NotFound = NotFound[:len(NotFound)-1]
    if len(DatabaseError) > 0: DatabaseError = DatabaseError[:len(DatabaseError) - 1]
    if len(ImgProcFailed) > 0: ImgProcFailed = ImgProcFailed[:len(ImgProcFailed) - 1]

    str_res = ''
    str_res += "<h3>Converting the images to dataset finished!!!</h3>"
    str_res += "<h4>Total files : {} </h4>".format(NN)
    str_res += "<h4>Number of Success : {} </h4>".format(nSucess)
    str_res += "<h4>Number of Image process failed : {} </h4>".format(nImgProcFailed)
    str_res += "<h6>{}</h6>".format(ImgProcFailed)
    str_res += "<h4>Number of Image not found : {} </h4>".format(nNotFound)
    str_res += "<h6>{} </h6>".format(NotFound)
    str_res += "<h4>Number of database save failed : {} </h4>".format(nDatabaseError)
    str_res += "<h6>{} </h6>".format(DatabaseError)

    # record the log file
    #with open(g_log_path, 'w') as f:
    #    f.writelines(str_res)

    return SAVE_OK, str_res

def search_db(img_name, img_path, startidx, search_method=0):
    global proc_rate
    proc_rate = 0

    delete_files_in_folder(static_path)
    delete_files_in_folder(res_path)

    img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_GRAYSCALE)
    if np.any(img) == None:
        return IMG_NOT_EXIST, None, None, None, None

    (h, w) = img.shape
    img = cv2.resize(img, ((int)(w/2), (int)(h/2)))

    try:
        c_cont, c_rate, c_hu, c_norm = get_edges_canny(img, False, False)
    except Exception:
        return IMG_PROC_ERROR, None, None, None, None

    """try:
        files = [name for name in os.listdir(res_path)]
    except:
        return RES_FOLDER_ERROR, None, None, None, None"""

    #img_list = ImageData.objects.all()
    start_rate = c_rate / min_rate
    end_rate = c_rate * min_rate
    img_list = ImageData.objects.filter(rate__gt = start_rate, rate__lt = end_rate)
    dists = []
    dists2 = []
    rates = []
    files = []
    NN = img_list.count()
    nn = 0

    for img_db in img_list.iterator(100):
        fname = img_db.name
        fpath = img_db.path
        rate = img_db.rate

        blob = img_db.get_data()
        cont = blob['cont']
        norm = blob['norm']


        # calculate distance
        rate_dist = calc_dist(c_rate, rate, 'rate')
        cont_dist = cv2.matchShapes(c_cont, cont, 1, 0.0)
        area_dist = area_diff(c_norm, norm)

        files.append([fpath, fname])
        rates.append(rate_dist)
        if search_method == 0:
            dists.append(cont_dist)
            dists2.append(area_dist)
        elif search_method == 1:
            dists.append(area_dist)
            dists2.append(cont_dist)

        nn += 1
        proc_rate = nn * 100 / NN

    nrates, ndists, ndists2, nfiles = get_filtered(rates, dists, dists2, files)
    res = get_similar_names(nrates, ndists, ndists2, nfiles, NResult)
    idx = 0

    r_scores = []
    r_scores2 = []
    r_files = []

    fnames_not_org = ''
    n_not_org = 0

    for each in res:
        r_scores.append(each[1])
        r_scores2.append(each[4])
        r_files.append(each[2])


        img = cv2.imread(os.path.join(each[2][0], each[2][1]), cv2.IMREAD_GRAYSCALE)
        if np.any(img) == None:
            img = cv2.imread(no_img_path, cv2.IMREAD_GRAYSCALE)
            n_not_org += 1
            fnames_not_org += each[2][1] + ','

        cv2.imwrite(os.path.join(res_path, each[2][1]), img)
        cv2.imwrite('{}{}.jpg'.format(static_path, idx + startidx), img)
        idx += 1

    if len(fnames_not_org) > 0: fnames_not_org = fnames_not_org[:len(fnames_not_org)-1]

    str_res = ''
    str_res += "<h3>Search finished in the dataset!!!</h3>"
    str_res += "<h4>Number of images searched: {} </h4>".format(NN)
    str_res += "<h4>Scaled rate deviation : {} </h4>".format(min_rate)
    if search_method == 0:
        str_res += "<h4>Search method : Contour Search </h4>"
    else:
        str_res += "<h4>Search method : Area Search </h4>"
    str_res += "<h4>Result number of Image searched : {} </h4>".format(idx)
    str_res += "<h4>Images that have not source : {} </h4>".format(n_not_org)
    str_res += "<h6>{}</h6>".format(fnames_not_org)

    proc_rate = 100
    return len(res), r_scores, r_scores2, r_files, str_res

def delete_files_in_folder(path):
    try:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        print(e)
        return -1
    return 0

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

def get_filtered(rates, score, score2, names):
    n = len(rates)
    n_score = []
    n_score2 = []
    n_names = []
    n_rates = []
    for i in range(n):
        if rates[i] < min_rate:
            n_score.append(score[i])
            n_score2.append(score2[i])
            n_names.append(names[i])
            n_rates.append(rates[i])

    return n_rates, n_score, n_score2, n_names

def get_similar_names(rates, dists, dists2, names, n=NResult):
    n = min(n, len(dists))
    dists = np.array(dists)
    res = []
    for i in range(n):
        idx = np.argmin(dists)
        res.append((rates[idx], dists[idx], names[idx], i + 1, dists2[idx]))
        dists[idx] = 100000000.0
    return res

"""
def search_engine(startidx, search_method=0):
    global proc_rate

    HState = delete_files_in_folder(res_path)
    if HState != 0:
        return HState, None, None, None
    HState = delete_files_in_folder(static_path)
    if HState != 0:
        return HState, None, None, None

    img = cv2.imread(criteria_picture, cv2.IMREAD_GRAYSCALE)
    if np.any(img) == None:
        return -2, None, None, None

    (h, w) = img.shape
    img = cv2.resize(img, ((int)(w / 2), (int)(h / 2)))

    cont_criteria, rate_criteria, hu_criteria, im_criteria = get_edges_canny(img, True, False)


    # -------------------------------------------------------------------------
    print ('cont information')

    value = {'cont' : cont_criteria, 'rate' : rate_criteria, 'norm' : im_criteria}
    try:
        foo = ImageData()
        foo.set_data(value)
        foo.save()
    except Exception:
        print ('error -------')
    
    print (foo.get_data())

    # -------------------------------------------------------------------------

    files = []
    try:
        for epath in path:
            for name in os.listdir(epath):
                files.append([epath, name])
    except Exception:
        return -3, None, None, None

    dists = []
    dists2 = []
    rates = []

    NN = len(files)
    nn = 0

    for each in files:
        file = each[1]
        epath = each[0]

        file_path = epath + file
        if DEBUG_MODE:
            print ('-- file name : {}'.format(file))
        cimg = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if np.any(cimg) == None:
            continue
        (h, w) = cimg.shape
        h = (int)(h / 2)
        w = (int)(w / 2)
        cimg = cv2.resize(cimg, (w, h))

        try:
            cont, ratev, hu_m, im = get_edges_canny(cimg)

            rate_dist = calc_dist(rate_criteria, ratev, 'rate')
            cont_dist = cv2.matchShapes(cont_criteria, cont, 1, 0.0)
            if 0:
                hu_dist = calc_dist(hu_criteria, hu_m, 'moment')
            im_dist1 =area_diff(im_criteria, im)
            #im_dist2 = area_diff_segment(im_criteria, im)
        except Exception:
            rate_dist = 9999.9
            cont_dist = 9999.9
            im_dist1 = 9999.9


        if DEBUG_MODE:
            print('{} : {}, {}, {}'.format(file, rate_dist, cont_dist, im_dist1))

        if search_method == 0:
            dists.append(cont_dist)
            dists2.append(im_dist1)
        elif search_method == 1:
            dists.append(im_dist1)
            dists2.append(cont_dist)


        nn += 1
        proc_rate = nn * 100 / NN
        if DEBUG_MODE:
            print (proc_rate)

        rates.append(rate_dist)

    nrates, ndists, ndists2, nfiles = get_filtered(rates, dists, dists2, files)
    res = get_similar_names(nrates, ndists, ndists2, nfiles, NResult)
    idx = 0

    r_scores = []
    r_scores2 = []
    r_files = []

    for each in res:
        r_scores.append(each[1])
        r_scores2.append(each[4])
        r_files.append(each[2])

        file_path = each[2][0] + each[2][1]
        img = cv2.imread(file_path)
        cv2.imwrite(res_path + each[2][1], img)
        cv2.imwrite('{}{}.jpg'.format(static_path, idx + startidx), img)
        idx += 1

    proc_rate = 100


    return len(res), r_scores, r_scores2, r_files"""


#------------------------------------------ for image folder to database converting backend process ---------------------------------------------------

def remove_db_record_period(cur_date, period):
    prev_date = cur_date + relativedelta(days = -period)

    del_records = list(ImageData.objects.filter(date__gte = prev_date))
    for each in del_records:
        each.delete()

    return len(del_records)

def save_img_folders_feature_backend(pathes, cur_datetime, period):
    global proc_rate
    proc_rate = 0

    print ('--------- converting image folder to directory started ------------------------------------')

    n_deleted = remove_db_record_period(datetime.date(cur_datetime.year, cur_datetime.month, cur_datetime.day), period)

    files = []
    try:
        for fpath in pathes:
            for fname in os.listdir(fpath):
                full_path = os.path.join(fpath, fname)
                file_ctime = os.path.getctime(full_path)
                file_datetime = datetime.datetime.fromtimestamp(file_ctime) # get file's timestamp
                delta_datetime = cur_datetime - file_datetime

                if delta_datetime.days > period:
                    continue
                files.append([fname, fpath, datetime.date(file_datetime.year, file_datetime.month, file_datetime.day)])

    except Exception:
        print ('--------------------------There is some invalid directory path ------------------------')
        for fpath in pathes:
            print (fpath)
        #return DIRECTORY_NOT_EXIST, "Invalid directories"

    NN = len(files)
    nn = 0
    nSucess = 0
    nNotFound = 0
    NotFound = ''
    nImgProcFailed = 0
    ImgProcFailed = ''
    nDatabaseError = 0
    DatabaseError = ''

    for fpair in files:
        fname = fpair[0]
        fpath = fpair[1]
        fdate = fpair[2]
        res_code = save_one_image_feature(fname, fpath, fdate)
        if res_code == SAVE_OK: nSucess += 1
        elif res_code == IMG_PROC_ERROR:
            nImgProcFailed += 1
            ImgProcFailed += fname + ','
        elif res_code == DATABASE_ERROR:
            nDatabaseError += 1
            DatabaseError += fname + ','
        elif res_code == IMG_NOT_EXIST:
            nNotFound += 1
            NotFound += fname + ','

        nn += 1
        proc_rate = nn * 100 / NN

    if len(NotFound) > 0: NotFound = NotFound[:len(NotFound)-1]
    if len(DatabaseError) > 0: DatabaseError = DatabaseError[:len(DatabaseError) - 1]
    if len(ImgProcFailed) > 0: ImgProcFailed = ImgProcFailed[:len(ImgProcFailed) - 1]

    str_res = ''
    str_res += '{}\n'.format(datetime.datetime.now())
    str_res += "------------Converting the images to dataset finished!!! ------------------\n\n\n"
    str_res += "removed from database : {}\n".format(n_deleted)
    str_res += "Total files to convert: {} \n".format(NN)
    str_res += "Number of Success : {} \n".format(nSucess)
    str_res += "Number of Image process failed : {} \n".format(nImgProcFailed)
    str_res += "{}\n".format(ImgProcFailed)
    str_res += "Number of fils that does not image : {} \n".format(nNotFound)
    str_res += "{} \n".format(NotFound)
    str_res += "Number of database save failed : {} \n".format(nDatabaseError)
    str_res += "{} \n\n\n".format(DatabaseError)

    # record the log file
    with open(g_log_path, 'a') as f:
        f.writelines(str_res)

    print ('------------converting image folder to directory finished--------------------')

    return SAVE_OK, str_res

def config_thread():
    global g_save_directory
    global g_capture_directory

    print ('------------------------config thread started-----------------------------')
    prev = datetime.datetime.now()

    while True:

        cur = datetime.datetime.now()
        cur_date = cur.day
        cur_hour = cur.hour
        cur_min = cur.minute

        admin_data = ParseAdminConfig(admin_file_path)

        interval = admin_data['interval']
        t_hour = admin_data['hour']
        t_minute = admin_data['minute']
        period = admin_data['period']
        directories = admin_data['directory']
        save_directory = admin_data['save_directory']
        capture_directory = admin_data['capture_directory']
        g_save_directory = save_directory
        g_capture_directory = capture_directory

        if (t_hour != cur_hour or t_minute != cur_min):
            time.sleep(30)
            continue

        delta_now = cur - prev
        if (delta_now.days < interval):
            time.sleep(30)
            continue

        save_img_folders_feature_backend(directories, cur, period)
        prev = cur
        time.sleep(60)

admin_config_thread = Thread(target=config_thread)
admin_config_thread.start()

#------------------------------------------             end         ---------------------------------------------------

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        #self.video = cv2.VideoCapture(CAMERA_INDEX)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        if TEST_MODE == True:
            # self.video = cv2.VideoCapture('static/1.mp4')
            self.video = 0
        else:
            self.video = cv2.VideoCapture(CAMERA_INDEX)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        if TEST_MODE == False:
            success, image = self.video.read()
        else:
            image =cv2.imread(TEST_IMG)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        cv2.imwrite(os.path.join(cap_img_path, cap_img_img), image)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

def gen(camera):
    global bCaptured

    while True:
        if not bCaptured:
            frame = camera.get_frame()
        else:
            frame = cv2.imread(os.path.join(cap_img_path, cap_img_img))
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def read_file(file_path):
    lines = []
    with open(file_path, 'r') as f:
        while True:
            cur = f.readline()
            if not cur: break
            lines.append(cur)
    if len(lines) != 3:
        return {'search_method' : 1, 'scale_range' : 20, 'res_number' : 100}

    res = {
        'search_method' : lines[0],
        'scale_range' : atoi(lines[1]),
        'res_number' : lines[2]
    }
    return res

def write_file(file_name, search_method, min_rate, nres):
    with open(file_name, 'w') as f:
        f.writelines(search_method)
        f.writelines('\n')
        f.writelines(min_rate)
        f.writelines('\n')
        f.writelines(nres)

def indexView(request):
    param = {'save_directory' : g_save_directory, 'capture_directory' : g_capture_directory}
    return render(request, "index.html", param)

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),content_type="multipart/x-mixed-replace; boundary=frame")

def camera(request):
    global bCaptured
    bCaptured = False
    return HttpResponse('Camera')

def capture(request):
    global bCaptured
    bCaptured = True
    return HttpResponse('Captured')

def global_param(data):
    global g_save_directory

    res_path = g_save_directory

    min_rate = data['min_rate']
    nres = atoi(data['nres'])

    global_param_setting(res_path, min_rate, nres)

def search(request):
    global gCount
    data = json.loads(request.POST['data'])

    if DEBUG_MODE:
        print ('---search called---')
    global_param(data)
    search_method = data['search_method']

    #n, scores, score2, files = search_engine(gCount, atoi(search_method))
    #if TEST_MODE == True:
    #    n, scores, score2, files, res_str = search_db(cap_img_img2, cap_img_path2, gCount, atoi(search_method))
    #else:
    n, scores, score2, files, res_str = search_db(cap_img_img, cap_img_path, gCount, atoi(search_method))

    if n == IMG_NOT_EXIST:
        param = {"s": -1, "n": n, "scores": None, "scores2": None, "files": None, 'res_str': res_str}
        return HttpResponse(json.dumps(param))
    elif n == IMG_PROC_ERROR:
        param = {"s": -1, "n": n, "scores": None, "scores2": None, "files": None, 'res_str': res_str}
        return HttpResponse(json.dumps(param))

    param = {"s" : gCount, "n" : n, "scores":scores, "scores2": score2, "files":files, 'res_str': res_str}
    gCount += n
    return HttpResponse(json.dumps(param))

def progress(request):
    global proc_rate
    if DEBUG_MODE:
        print ('---------------------------------------------------------------------')
        print (proc_rate)
    return HttpResponse((int)(proc_rate))

def get_tmpl_names(request):
    tmpl_folder = "static/ImageRetrieval/tmpls/"
    files =[name for name in os.listdir(tmpl_folder)]
    param = {'list' : files}
    return HttpResponse(json.dumps(param))

def get_file_info(request):
    file_path = request.POST['name']
    param = read_file(file_path)
    return HttpResponse(json.dumps(param))

def set_file_info(request):
    fname = request.POST['filename']
    search_method = request.POST['search_method']
    min_rate = request.POST['min_rate']
    nres = request.POST['nres']

    write_file(fname, search_method, min_rate, nres)
    return HttpResponse('OK')

def convert_directories(request):
    data = json.loads(request.POST['data'])
    search_path = data['search_path']
    fpathes = []
    for each in search_path:
        if each == None or each == '':
            continue
        if each[-1] != '/':
            each = each + '/'
            fpathes.append(each)

    res_code, str_res = save_img_folders_feature(fpathes)
    return HttpResponse(json.dumps({'res_code' : res_code, 'str_res' : str_res}))

def save_captured_img(request):
    global bCaptured
    global g_capture_directory

    if not bCaptured:
        return HttpResponse("Please capture image first")

    img = cv2.imread(os.path.join(cap_img_path, cap_img_img), cv2.IMREAD_GRAYSCALE)
    if np.any(img) == None:
        return HttpResponse("Can't get captured image")


    dtime = datetime.datetime.now()
    dtime = '{}'.format(dtime)
    dtime = dtime.replace('-', '')
    dtime = dtime.replace(':', '')
    dtime = dtime.replace(' ', '')
    dtime = dtime.replace('.', '')

    fname = request.POST['fname']
    fname = '{}_{}.jpg'.format(fname, dtime)
    #fpath = request.POST['fpath']
    fpath = g_capture_directory

    try:
        names = [name for name in os.listdir(fpath)]
    except Exception:
        return HttpResponse("image path does not exist")

    cv2.imwrite(os.path.join(fpath, fname), img)

    res_code = save_one_image_feature(fname, fpath)
    if res_code == SAVE_OK:
        return HttpResponse("Sucessfully saved")
    elif res_code == IMG_PROC_ERROR:
        return HttpResponse("Please capture the proper image")
    elif res_code == DATABASE_ERROR:
        return HttpResponse("Can not save to the database")
    elif res_code == IMG_NOT_EXIST:
        return HttpResponse("Can not find the captured image")
    else:
        return HttpResponse("Unknown error")
