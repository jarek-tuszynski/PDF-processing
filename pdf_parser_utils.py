# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 2022

@author: Jarek Tuszynski

Set of utilities to build on "PyMuPDF"
"""
import os
import random
import fitz  # load PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from difflib import SequenceMatcher as SM
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
def find_lines(img, axis: int = 0, plots: bool = False):
    # img1 = ImageOps.grayscale(img)
    # mat  = np.array(img1).T # transpose to keep the same shape
    #gray = 255-cv2.cvtColor(mat,cv2.COLOR_BGR2GRAY)
    gray = 1-img/255
    r_mean = np.mean(gray, axis=axis)
    r_mean = np.convolve(r_mean,[1,1,1,1,1], 'same')/5
    rows = np.where(r_mean > 0.5)[0]
    rows = rows[np.diff(rows, append=1e6)>1]
    if plots:
        mat = np.array(img)
        nc = mat.shape[0]
        for row in rows:
            cv2.line(mat,(0,row),(nc,row),(255,0,0),5)
        cv2.imwrite("lines.jpg", mat)

        # Plot the histogram
        plt.hist(r_mean, bins=100)
        plt.savefig('hist.png')

        plt.plot(r_mean)
        plt.savefig('r_mean.png')
    return rows

# ---------------------------------------------------------------------------
def get_cell_borders(img, bbox0, thresh = [0.7, 1.3]):
    '''
    locate extend of a cell in the form based on the image and a text box inside.
    The approach finds vertical and horizontal lines in the image.

    Parameters
    ----------
    img : numpy / openCV image - image of the page
    bbox0 : array of 4 integers - bounding-box of text-box of interest
    thresh : array of 2 floats - min and max of the box size ratios text-box / cell-box
         The default is [0.7, 1.3].

    Returns
    -------
    bbox : array of 4 integers - bounding box of the cell boundaries

    '''
    l0, t0, r0, b0 = bbox0
    min_row = (t0+b0)//2
    bbox_area0 = (r0-l0)*(b0-t0)
    hbars = find_lines(img, 1)
    if hbars[-1]<b0:
        return None

    # find index of the horizontal bar above the anchor bbox (min_row)
    line_offset = max(0, np.argmax(hbars>min_row) -1)
    y1 = hbars[line_offset]
    y2 = hbars[line_offset+1]
    #line_img = img.crop((0, y1, img.size[0], y2))
    line_img = img[y1:y2,:]
    vbars = find_lines(line_img, 0)
    nbox  = len(vbars)-1
    for ibox in range(nbox):
        x1 = vbars[ibox]
        x2 = vbars[ibox+1]
        bbox = [x1, y1, x2, y2]
        bbox_area1 = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
        q = bbox_area1/bbox_area0
        if q>thresh[0] and q<thresh[1]: # if expected and observed bbox are similar size then accept
            return bbox

    return None


# ---------------------------------------------------------------------------
def get_cell_fflood(img, bbox0, thresh = [0.7, 1.3], debug_fname=None):
    '''
    locate extend of a cell in the form based on the image and a text box inside.
    The approach uses flood-fill

    Parameters
    ----------
    img : numpy / openCV image - image of the page
    bbox0 : array of 4 integers - bounding-box of text-box of interest
    thresh : array of 2 floats - min and max of the box size ratios text-box / cell-box
         The default is [0.7, 1.3].
    debug_fname - file name of the debbuging image (optional)

    Returns
    -------
    bbox : array of 4 integers - bounding box of the cell boundaries
    '''
    def rand_range(a,b,k):
        # random integer in between a and b with (a-b)/k margin on both ends
        d = abs(a-b) // k
        if a+d>=b-d:
            return (a+b)//2
        return random.randint(a+d, b-d)

    val = 127
    height, width = img.shape
    if bbox0[0]>width or bbox0[1]>height:
        return None
    bbox0[2] = min(bbox0[2], width)
    bbox0[3] = min(bbox0[3], height)
    debug = (debug_fname is not None)
    if debug:
        img1 = Image.fromarray(img).convert('RGB')
        d = ImageDraw.Draw(img1)
        d.rectangle(bbox0, fill =None, outline ="green", width=3)
        #clr = ['red', 'purple', 'maroon', 'darkred', 'indianred']
        clr = ['red', 'cyan', 'magenta', 'blue', 'yellow']
        font = ImageFont.truetype("arial.ttf", 20)

    #l, t, r, b = bbox0
    bbox_area0 = (bbox0[3]-bbox0[1])*(bbox0[2]-bbox0[0])
    # multiple tries just in case we randomly hit a center of "O", etc. on the first try
    for i in range(20):
        img0 = img.copy()
        # pick a random seed point not too cloase to the edge
        x = rand_range(bbox0[0], bbox0[2], 4)
        y = rand_range(bbox0[1], bbox0[3], 4)
        if img[y,x] == 255:

            cv2.floodFill(img0, None, (x,y), val)
            rows, cols = np.where(img0 == val)
            bbox = [np.min(cols), np.min(rows), np.max(cols), np.max(rows)]
            if debug and i<=4:
                d.rectangle(bbox, fill =None, outline =clr[i], width=3)
                d.text((x,y), 'X', fill=clr[i], font=font)
            bbox_area1 = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
            q = bbox_area1/bbox_area0
            #print(i, x, bbox_area0, bbox_area1)
            if q>thresh[0] and q<thresh[1]: # if expected and observed bbox are similar size then accept
                if debug:
                    img1.save(debug_fname)
                return bbox

    if debug:
        img1.save(debug_fname)
    return None

# ------------------------------------------------------------------------
def save_pdf_pages(pdf_fname: str, path_out: str, overlay_text: str = 'none'):
    '''
    extract all pages from a pdf file

    Inputs:

    pdf_fname: (string) input PDF filename
    path_out:  (string) full path to output directory
    overlay_text: string: what to do with text layer:
        'none' : do not show text layer
        'flat' : show text layer as a flat structure
        'hierarchical' : show text layer grouped into lines and blocks
    '''
    doc = fitz.open(pdf_fname)  # open document
    if not os.path.exists(path_out): # if output folder does not exist -> create it
        os.makedirs(path_out)

    # loop over all the pages
    for page_num in range(len(doc)):
        fname_out = os.path.join(path_out, 'page_{:03d}.png'.format(page_num))
        extract_pdf_page(doc, page_num, overlay_text).save(fname_out)

# ------------------------------------------------------------------------
def extract_pdf_page(doc, page_number: int, overlay_text: str = 'hierarchical'):
    '''
    Create pillow image showing underlying image and scanned text

    Parameters
    ----------
    doc: input document PDF object (output of "fitz.open(fname_in)") or a filename sting
    overlay_text: string: what to do with text layer:
        'none' : do not show text layer
        'flat' : show text layer as a flat structure
        'hierarchical' : show text layer grouped into lines and blocks

    Returns
    -------
    Pillow image

    '''
    if isinstance(doc, str): # doc is a string so open this document
        doc = fitz.open(doc) # now a fitz.Document class

    page = doc[page_number] # now a fitz.Page class
    pix  = extract_pdf_image(page, False)
    if overlay_text == None or overlay_text == 'none':
        txt = None
    elif overlay_text == 'flat':
        txt  = extract_pdf_text(page, False)
    elif overlay_text == 'hierarchical':
        txt  = extract_pdf_text(page, True)
    return draw_image_text( pixelmap2pil(pix) , txt)

# ------------------------------------------------------------------------
def extract_pdf_image(page, verbose: bool = False):
    '''
    extract image of a given page
    This approach seems to extract a single image per page, it is probably a
           rentering of all the image and text layers stored in PDF file
           for a given page. It includes margins, etc.

    Inputs:
        page: fitz.Page class
        verbose: (boolean) print any image info?
    Output:
        pix : PDF pixmap object for alg=0 or array of them for alg=1
    '''
    blocks = page.get_text("dict")["blocks"]
    _, dpi = get_text_scale(blocks)
    pix = page.get_pixmap(dpi=dpi)  # render page to an image

    if verbose:
        print('   -> Pixmap:     size: ({} x {}); dpi: {}'.format( pix.width, pix.height, dpi))

    return pix

# ------------------------------------------------------------------------
def extract_raw_pdf_images(doc, page_number: int, verbose: bool = False):
    '''
    extract images on a given page.
    This approach seems to access lower level underlying data and get the original
           images stored in PDF. There might be multiple images per page
           and they can be rotated.

    Inputs:

        doc: input document PDF object (output of "fitz.open(fname_in)") or a filename sting
        page_number: (int) page number
        verbose: (boolean) print any image info?
    Output:
        pix : array of PDF pixmap objects
    '''
    if isinstance(doc, str): # doc is a string so open this document
        doc = fitz.open(doc) # now a fitz.Document class

    pix_array = []
    for img in doc.get_page_images(page_number):
        (xref, smask, width, height, bpc, colorspace, altcolorspace, name, filtr) = img
        pix = fitz.Pixmap(doc, xref)
        pix_array.append( pix )
        if verbose:
            print('   -> page image: size: ({} x {}); filter: {}; xref: {}'.format( width, height, filtr, xref))
            print('   -> Pixmap:     size: ({} x {})'.format( pix.width, pix.height))

    return pix_array

# ------------------------------------------------------------------------
def extract_pdf_text(page, flat : bool = True):
    '''
    extract text layer of a given page

    Inputs:
        page: fitz.Page class
        flat: (Boolean) return flat list of text boxes or hierarhical structure
    Output:
        text : array of dict - where each dict has "text" and "bbox" field.
            bbox is in an array using left-top-right-bottom order. The
            coordinates are pixels of the image

            If "flat" is True then "text" field will be an array of other dicts
            allowing groupings of text boxes into lines and then blocks
    '''
    blocks = page.get_text("dict")["blocks"]
    scale, _  = get_text_scale(blocks) # determine scaling needed to use pixel coordinates

    # extract text boxes
    flat_text = []
    hier_text = []

    if not flat:
        for block in blocks:
            if block["type"] == 0: # textbox block
                text1 = []
                for line in block['lines']:
                    text2 = []
                    for span in line['spans']:
                        tbox = {'text': span['text'], 'bbox': bbox_scale(span['bbox'], scale)}
                        #flat_text.append(tbox)
                        text2.append(tbox)
                    tbox = {'text': text2, 'bbox': bbox_scale(line['bbox'], scale, 1)}
                    text1.append(tbox)
                tbox = {'text': text1, 'bbox': bbox_scale(block['bbox'], scale, 2)}
                hier_text.append(tbox)
    else:
        words = page.get_text("words")
        for arr in words:
            x0, y0, x1, y1, word, block_no, line_no, word_no = arr
            tbox = {
                'text': word,
                'bbox': bbox_scale([x0, y0, x1, y1], scale),
                'line_no': line_no,
                'word_no': word_no
                }
            flat_text.append(tbox)

    return flat_text if flat else hier_text

# ------------------------------------------------------------------------
def get_text_scale(blocks, verbose: bool = False):
    '''
    Get scale factor needed to convert text bounding boxes from native units to pixels

    Parameters
    ----------
    blocks : output of page.get_text("dict")["blocks"]
    verbose: (boolean) print any image info?

    Returns
    -------
    scale : [sx, sy]
    dpi   : image scaling "dots per inch"

    '''
    # determine scaling needed to use pixel coordinates
    scale = [1, 1] # default
    dpi   = 300
    for block in blocks:
        if block["type"] == 1: # image block
            bbox = block['bbox']
            dx = block['width' ]/(bbox[2]-bbox[0])
            dy = block['height']/(bbox[3]-bbox[1])
            dpi = int(block['width']/8.5) # 8.5 inch is a page width
            scale = [dx, dy]
            if verbose:
                print(bbox, [block['width'], block['height']],  '->', scale, dpi)
            break
    return scale, dpi

# ------------------------------------------------------------------------
def draw_image_text(img, text):
    '''
    Create pillow image showing underlying image and scanned text

    Parameters
    ----------
    img  : pixmap - output of extract_pdf_image or PIL.Image
    text : array of dicts - from extract_pdf_text

    Returns
    -------
    Pillow image

    '''
    #img = pixelmap2pil(img)
    if text == None:
        return img
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)
    for tbox in text:
        bbox = tbox['bbox']
        text = tbox['text']
        if isinstance(text, str): # flat data structure
            d.text(bbox[:2], text, fill="red", font=font)
            d.rectangle(bbox, fill =None, outline ="green")
        else:
            d.rectangle(bbox, fill =None, outline ="magenta")
            for line in text:
                d.rectangle(bbox, fill =None, outline ="cyan")
                for span in line['text']:
                    d.rectangle(bbox, fill =None, outline ="green")
                    d.text(bbox[:2], span['text'], fill="red", font=font)

    # hbars = find_lines(img)
    # for row in hbars:
    #     d.line((0,row,img.size[0],row), fill ="magenta")
    return img

# ------------------------------------------------------------------------
def bbox_scale(bbox, scale, enlarge: int = 0):
    '''
    rescale bounding box and convert to integer pixel coordinates

    Parameters
    ----------
    bbox : array of 4 floats -  left-top-right-bottom
    scale : array of 4 floats - [scale_x, scale_y]
    enlarge : integer - make the box a few pixels larger

    Returns
    -------
    array of 4 integers -  left-top-right-bottom

    '''
    l, t, r, b = bbox
    dx, dy = scale
    l = max(0, int(l*dx - enlarge))
    r = max(0, int(r*dx - enlarge))
    t = int(t*dy + enlarge)
    b = int(b*dy + enlarge)
    return [l, t, r, b]

# ------------------------------------------------------------------------
def pixelmap2pil(pix):
    '''
    Image format conversion

    Parameters
    ----------
    pix : (fritz.Pixelmap) image in PyMuPDF internal format

    Returns
    -------
    img : (PIL.Image) image converted to Pillow's Image format

    '''
    cspace = pix.colorspace
    if cspace is None:
        mode = "L"
    elif cspace.n == 1:
        mode = "L" if pix.alpha == 0 else "LA"
    elif cspace.n == 3:
        mode = "RGB" if pix.alpha == 0 else "RGBA"
    else:
        mode = "CMYK"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img

# ------------------------------------------------------------------------
def fuzzy_match(src, trgs):
    match = 0
    for trg in trgs:
       match = max(match,  SM(None, src.lower(), trg.lower()).ratio())
    return match