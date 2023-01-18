# -*- coding: utf-8 -*-

# Python program to extract text from all the images in a folder
# storing the text in corresponding files in a different folder
#from PIL import Image
import pytesseract as pt
import os
import numpy as np
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import fitz  # load PyMuPDF

# importing local modules
import pdf_parser_utils as ppu
import pdf_parser_forms as ppf

#=============================================================================
def find_chip(results, image, pattern):
    ''' look  through "results" of ocr and find strings with phrase matching
        pattern string. return "image" segment ("chip") with the tightest fit.
    '''
    chip_size = 1e6
    for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x1 = results["left"][i]
        y1 = results["top"][i]
        x2 = results["width"][i] + x1
        y2 = results["height"][i] + y1
        text = results["text"][i]
        csize = (x2-x1)*(y2-y1)
        if text == pattern and csize<chip_size:
            chip = image[y1:y2, x1:x2]
            chip_size = csize
            #print (csize)

    return chip.copy()

#=============================================================================
def find_chips(fname, patterns):
    image = cv2.imread(fname)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # perform ocr using pytesseract for python
    results = pt.image_to_data(gray, output_type=Output.DICT)

    # find image chips with key words
    chips = {}
    for pattern in patterns:
        chips[pattern] = find_chip(results, gray, pattern)

    return chips

#=============================================================================
def mark_page(results, image):
    min_conf = 80
    for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # extract the OCR text itself along with the confidence of the
        # text localization
        text = results["text"][i]
        conf = int(results["conf"][i])
        # filter out weak confidence text localizations
        if conf > min_conf:
            # display the confidence and text to our terminal
            print("Confidence: ​​", conf, " Text: ", text)

            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw a bounding box around the text along
            # with the text itself
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)



#=============================================================================
def mark_chips(res, chip, color, image1, image2):
     i_chip = np.where( np.array(color) > 0)[0][0]
     mask = np.zeros(image2.shape, dtype=bool)

     # perform connected components analysis
     threshold = 0.7
     msk = np.array((res >= threshold), dtype="uint8")
     output = cv2.connectedComponentsWithStats(msk, 4, cv2.CV_32S)
     centroids = output[3].round().astype(int) # The fourth cell is the centroid matrix
     centroids = centroids[1:,:] # remove background class

     # create chip mask
     kernel = np.ones((3, 3), dtype="uint8")
     msk = np.array(chip<127, dtype="uint8")
     msk = cv2.dilate(msk, kernel, iterations=1)

     h, w = chip.shape
     i_box = 0
     for i_box, (l, t) in enumerate(centroids):
         # if i_box==0:
         #     continue
         b,r = t+h, l+w
         cv2.rectangle(image1, (l, t), (r, b), color, 3)
         cv2.putText(image1, str(i_box), (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

         mask[t:b,l:r] = True

         img = image2[t:b,l:r]
         img1 = img.copy()
         img[msk>0] = 255
         image2[t:b,l:r] = img
         cv2.imwrite("chip_{:02}_{:02}.png".format(i_chip, i_box), np.vstack((chip, img1, img)))
         print ('mark_chips', i_chip, i_box, b, r)

     return mask

#=============================================================================
def parse_marking(row, num):
    msk = np.array(row < 127, dtype="uint8")
    n_label, labels, stats, centroids = cv2.connectedComponentsWithStats(msk, 4, cv2.CV_32S)
    sizes = stats[1:, -1]
    largest = np.argmax(sizes)+1
    x = centroids[largest, 0]
    d = row.shape[1]//3 # divide row into 3 side by side regions
    k = int(x/d)        # which one is the centroid of the largest object in

    # plot results
    rgb1 = np.dstack((row, row, row))
    img1 = np.array(255*labels/n_label, dtype="uint8")
    img3 = cv2.applyColorMap(img1, 4)
    img3[labels==0] = (255,255,255)
    msk  = np.array(labels == largest, dtype="uint8")
    rgb2 = (1-np.dstack((msk, msk, msk)))*255
    cv2.imwrite("chip3_{:02}.png".format(num), np.vstack((rgb1, img3, rgb2)))

    return k

#=============================================================================
def find_check_boxes(gray, chips,  threshold1):
    df = pd.DataFrame([], columns= ['left', 'top', 'width', 'height', 'score1' , 'score2', 'score3'])
    for chip in chips:
        # locate chips on a page
        corr = cv2.matchTemplate(gray, chip, cv2.TM_CCOEFF_NORMED)

        # create chip mask
        kernel = np.ones((3, 3), dtype="uint8")
        chip_msk2 = np.array(chip<127, dtype="uint8")
        chip_msk3 = cv2.dilate(chip_msk2, kernel, iterations=1)

        # perform connected components analysis
        msk = np.array((corr >= threshold1/100), dtype="uint8")
        n_label, labels, stats, centroids = cv2.connectedComponentsWithStats(msk, 4, cv2.CV_32S)

        if n_label == 0:
            return df
        h, w = chip.shape
        for i_label in range(1,n_label):
            lab_msk = (labels == i_label)

            score1 = corr[lab_msk].max()
            t, l = np.where(lab_msk & (corr == score1))
            t, l = t[0], l[0]
            b, r = t+h, l+w

            chip_img = 1 - gray[t:b,l:r]/255
            score2 = chip_img[chip_msk2 >0].mean()
            score3 = chip_img[chip_msk3==0].mean()
            df.loc[len(df.index)] = [l,t,w,h, 100*score1, 100*score2, 100*score3]

    return df

#=============================================================================
def find_check_boxes2(gray, chips,  threshold1,  threshold2):
    df = pd.DataFrame([], columns= ['left', 'top', 'width', 'height', 'score1' , 'score2', 'score3'])
    for chip in chips:
        # locate chips on a page
        res1 = cv2.matchTemplate(gray, chip, cv2.TM_CCOEFF_NORMED)

        # create chip mask
        kernel = np.ones((3, 3), dtype="uint8")
        chip_msk1 = np.array(chip<127, dtype="uint8")
        chip_msk2 = cv2.dilate(chip_msk1, kernel, iterations=1)

        # perform connected components analysis
        msk = np.array((res1 >= threshold1), dtype="uint8")
        n_label, labels, stats, centroids = cv2.connectedComponentsWithStats(msk, 4, cv2.CV_32S)
        centroids = centroids.round().astype(int) # The fourth cell is the centroid matrix
        centroids = centroids[1:,:] # remove background class

        h, w = chip.shape
        for i_box, (l, t) in enumerate(centroids):
            i_label = i_box+1
            score1 = res1[labels == i_label].mean()
            l, t = int(round(l)), int(round(t))
            b, r = t+h, l+w

            chip_img = gray[t:b,l:r]
            score2 = chip_img[chip_msk1>0].mean()
            score2 = (1-score2/255)
            if score2>threshold2:
                chip_img[chip_msk2>0] = 255
                chip_msk3 = (chip_img<127)
                score3 = chip_msk3.mean()
                score3 = int(round(1000* score3))
                score1 = int(round(100 * score1))
                score2 = int(round(100 * score2))
                df.loc[len(df.index)] = [l,t,w,h,score1, score2, score3]

    return df

#=============================================================================
def check_boxes():
    # If you don't have tesseract executable in your PATH, include the following:
    pt.pytesseract.tesseract_cmd = r'C:\Users\tuszynskij\AppData\Local\Tesseract-OCR\tesseract.exe'

    # path for the folder for getting the raw images
    folder_in = r'C:\projects\CNS project\form-image-mgmt\input'
    folder_out= r'C:\projects\CNS project\form-image-mgmt\output_chip'

    words = ['cbox5'] #, 'cbox2', 'cbox4']
    color = [ (0, 0, 255), (255, 0, 0), (0, 255, 0) ]



    # load chips
    chips = []
    for key in words:
        img = cv2.imread("chips\chip_{}.png".format(key), cv2.IMREAD_GRAYSCALE)
        chips.append(img)

    fname = '56092562 EF-69 PM02 - eDCRO_337685'
    fname = '56096537 Hoist PM03 - eDCRO_337688'
    fname = '56092341 EMEX PM02 - eDCRO_337684'
    fname_in  = os.path.join(folder_in, fname+'.pdf')
    doc = fitz.open(fname_in)  # open document

    theshold = [55, 80, 1]

    n = len(doc)
    #n = 10
    for page_num in range(n):

        # fetch a page and a chip
        image = ppu.extract_pdf_page(fname_in, page_num, 'none')
        rgb   = np.array(image)
        gray  = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        df    = find_check_boxes(gray, chips, theshold[0])
        for irow, row in df.iterrows():
            l,t  = int(row['left']),  int(row['top'])
            w,h  = int(row['width']), int(row['height'])
            s1,s2,s3 = row['score1'], row['score2'], row['score3']
            b,r = t+h, l+w
            k = int(s2>theshold[1]) + int(s3>theshold[2])
            cv2.rectangle(rgb, (l, t), (r, b), color[k], 3)
            lab = '{:0.0f}_{:0.0f}_{:0.1f}'.format(s1, s2, s3)
            cv2.putText(rgb, lab, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color[k], 3)

        fname_out  = os.path.join(folder_out,  r'{}\page_{:02}.png'.format(fname, page_num))
        cv2.imwrite(fname_out, rgb)


#=============================================================================
def sat_unsat_na():
    # If you don't have tesseract executable in your PATH, include the following:
    pt.pytesseract.tesseract_cmd = r'C:\Users\tuszynskij\AppData\Local\Tesseract-OCR\tesseract.exe'

    # path for the folder for getting the raw images
    path = r'C:\projects\CNS project\form-image-mgmt\output\56092562 EF-69 PM02 - eDCRO_337685'

    words = ['SAT', 'UNSAT', 'NA']
    chip_source = 'detect' # 'load' vs. 'detect'
    # min_conf = 80
    color = [ (255, 0, 0), (0, 255, 0), (0, 0, 255)]


    # iterating the images inside the folder
    for imageName in os.listdir(path):
        if not imageName.endswith('_005.png'):
            continue
        inputPath = os.path.join(path, imageName)

        # get chips
        if chip_source == 'detect':
            chip = find_chips(inputPath, words)
            for key in chip:
                cv2.imwrite("chip_{}.png".format(key), chip[key])
        elif chip_source == 'load':
            chip = {}
            for key in words:
                chip[key] = cv2.imread("chip_{}.png".format(key), cv2.IMREAD_UNCHANGED)

        image = cv2.imread(inputPath)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # locate those chips in the image
        res = {}
        for key in chip:
            res[key] = cv2.matchTemplate(gray, chip[key], cv2.TM_CCOEFF_NORMED)


        # mark image regions with the chips
        rgb = image.copy()
        prof1 = mark_chips(res['SAT'  ], chip['SAT'  ], color[0], rgb, gray)
        prof2 = mark_chips(res['UNSAT'], chip['UNSAT'], color[1], rgb, gray)
        prof3 = mark_chips(res['NA'   ], chip['NA'   ], color[2], rgb, gray)
        rgb1   = np.dstack((prof1, prof2, prof3))*255
        cv2.imwrite("SAT_boxes3.png", rgb)
        cv2.imwrite("SAT_gray.png", gray)
        cv2.imwrite("SAT_msk.png", rgb1)

        # locate rows with 'SAT', 'UNSAT', 'NA' in that order
        rgb = np.dstack((gray, gray, gray))
        vprof = (prof1.sum(axis=1)>0) + (prof2.sum(axis=1)>0)*2 + (prof3.sum(axis=1)>0)*4
        labeled, nr_objects = ndimage.label(vprof == 7)
        print("Num obj:", nr_objects)
        for i_obj in range(nr_objects):
            vmsk = (labeled == i_obj+1)
            row = rgb[vmsk,:,:]
            msk1 = prof1[vmsk,:].sum(axis=0)>0
            msk2 = prof2[vmsk,:].sum(axis=0)>0
            msk3 = prof3[vmsk,:].sum(axis=0)>0

            y  = np.where(vmsk)[0]
            x1 = np.where(msk1)[0]
            x2 = np.where(msk2)[0]
            x3 = np.where(msk3)[0]
            h  = (y[-1]-y[0]+1)
            w2 = (x2[-1]-x2[0]+1)
            w3 = (x3[-1]-x3[0]+1)
            dx  = round(1.5*h)
            dy  = round(1.5*h)
            if  w2 == len(x2) and  w3 == len(x3) and \
                x1.mean()<x2.mean() and x2.mean()<x3.mean() :
                print('group:', i_obj, round(y.mean()), round(x1.mean()), round(x2.mean()), round(x3.mean()))
                cv2.rectangle(rgb, (x1[0]-dx, y[0]-dy ), (x3[-1]+dx, y[-1]+dy), (255, 0, 0), 2)
                row = gray[ (y[0]-dx) : (y[-1]+dy), (x1[0]-dy) : (x3[-1]+dx)]
                k = parse_marking(row, i_obj)
                cv2.putText(rgb, words[k], (x1[0], y[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


            else:
                print('group:', i_obj, round(y.mean()), 'bad' )

        cv2.imwrite("SAT_boxes1.png", rgb)


        plt.plot(vprof);  plt.show()
        plt.plot(labeled);  plt.show()

#=============================================================================
def main():
    #sat_unsat_na()
    check_boxes()

#=============================================================================
if __name__ == '__main__':
    main()