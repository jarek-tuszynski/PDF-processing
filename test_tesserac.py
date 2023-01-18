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
from scipy import ndimage

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
def main():
    # If you don't have tesseract executable in your PATH, include the following:
    pt.pytesseract.tesseract_cmd = r'C:\Users\tuszynskij\AppData\Local\Tesseract-OCR\tesseract.exe'

    # path for the folder for getting the raw images
    path = r'C:\projects\CNS project\form-image-mgmt\output\56092562 EF-69 PM02 - eDCRO_337685'

    words = ['SAT', 'UNSAT', 'NA']
    # min_conf = 80

    # iterating the images inside the folder
    for imageName in os.listdir(path):
        if not imageName.endswith('_004.png'):
            continue
        inputPath = os.path.join(path, imageName)
        image = cv2.imread(inputPath)
        rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # perform ocr using pytesseract for python
        results = pt.image_to_data(gray, output_type=Output.DICT)

        # find image chips with key words
        chip1 = find_chip(results, gray, "SAT")
        chip2 = find_chip(results, gray, "UNSAT")
        chip3 = find_chip(results, gray, "NA")

        # save them
        cv2.imwrite("chip_SAT.png", chip1)
        cv2.imwrite("chip_UNSAT.png", chip2)
        cv2.imwrite("chip_NA.png", chip3)

        # locate those chips in the image
        res1 = cv2.matchTemplate(gray, chip1, cv2.TM_CCOEFF_NORMED)
        res2 = cv2.matchTemplate(gray, chip2, cv2.TM_CCOEFF_NORMED)
        res3 = cv2.matchTemplate(gray, chip3, cv2.TM_CCOEFF_NORMED)

        # mark image regions with the chips
        rgb = image.copy()
        prof1 = mark_chips(res1, chip1, (255, 0, 0), rgb, gray)
        prof2 = mark_chips(res2, chip2, (0, 255, 0), rgb, gray)
        prof3 = mark_chips(res3, chip3, (0, 0, 255), rgb, gray)
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




        #cv2.imwrite(inputPath+".png", image)

        #cv2.imshow("Image", image)





        # for line in text.split('\n'):
        #     wcount = 0
        #     line = line.replace('  ',' ')
        #     for word in words:
        #         if (word+' ' in line) or (word+'/' in line) or (' '+word in line) or ('/'+word in line):
        #             wcount += 1

        #     if wcount>=3:

        #         case = ''
        #         line = line.replace(' ','')
        #         if 'SAT/UNSAT/NA' in line:
        #             case = 'none'
        #         elif ('NSAT/NA' in line) and ((not ' SAT' in line) or (not 'SAT/' in line)):
        #             case = 'SAT'
        #         elif ('SAT/UNSA' in line) and ((not 'NA ' in line) or (not '/NA' in line)):
        #             case = 'NA'
        #         else:
        #             match = re.search(r'SA.*NA', line)
        #             if match:
        #               case = 'UNSAT'

        #         print(case, line)

        a=1
        # fullTempPath = os.path.join(path, imageName+".txt")
        # print(imageName)

        # # saving the  text for every image in a separate .txt file
        # with open(fullTempPath, 'w+', encoding='utf-8') as f:
        #     f.write(text)

#=============================================================================
if __name__ == '__main__':
    main()