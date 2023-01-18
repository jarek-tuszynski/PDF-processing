# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:03:05 2022

@author: tuszynskij
"""
import numpy as np
from string import punctuation
from difflib import SequenceMatcher as SM
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

# importing local modules
import pdf_parser_utils as ppu

# -----------------------------------------------------------------------------
def bbox_offset(bbox, offset):
    '''
    offset bounding box bbox + origin

    Parameters
    ----------
    bbox : array of 4 numbers - bounding box
        DESCRIPTION.
    offset : array of 2 or more numbers - offset
        DESCRIPTION.

    Returns
    -------
    bbox : array of 4 numbers 

    '''
    return [bbox[0]+offset[0], bbox[1]+offset[1], 
            bbox[2]+offset[0], bbox[3]+offset[1]]

# -----------------------------------------------------------------------------
def bbox_inside(bbox_small, bbox_big, all_inside: bool = True):
    '''
    True or false: is the small bounding box inside big bounding box?

    Parameters
    ----------
    bbox_small : array of 4 numbers - bounding box
    bbox_big : array of 4 numbers - bounding box
    all_inside : bool, optional - (The default is True) if false than only the 
                center of the small box will be tested.

    Returns
    -------
    is the small bounding box inside big bounding box

    '''
    l0, t0, r0, b0 = bbox_small
    l1, t1, r1, b1 = bbox_big
    if all_inside:
        if l0>l1 and l0<r1 and t0>t1 and t0<b1 and r0>l1 and r0<r1 and b0>t1 and b0<b1:
            return True
    else:
        if l0>l1 and l0<r1 and t0>t1 and t0<b1:
            return True
        if l0>l1 and l0<r1 and b0>t1 and b0<b1:
            return True
        if r0>l1 and r0<r1 and b0>t1 and b0<b1:
            return True
        if r0>l1 and r0<r1 and t0>t1 and t0<b1:
            return True

    return False
    
#==============================================================================
class form_parser:
    ''' Class implementing form parsing
    '''
    def __init__(self, df):
        tbl_size = [0,0] # number of table rows and columns
        img_size = [0,0] # width and height in pixels

        df["bbox2"] = 0
        for irow, row in df.iterrows():
            r, c = eval(row['location'])
            x1, y1, x2, y2 = eval(row['bbox'])
            df.at[irow, 'location'] = (r,c)
            df.at[irow, 'bbox']     = (x1, y1, x2, y2)
            tbl_size = [max(tbl_size[0], r+1), max(tbl_size[1], c+1)]
            img_size = [max(tbl_size[0], x2),  max(tbl_size[1], y2)]
            
        df['quality'] = 0
        df["text"]    = ''
        df["value"]   = ''
        df["bbox2"]   = df["bbox"]
            
        self.data   = df.loc[df['field'] != 'ANCHOR']
        self.anchor = df.loc[df['field'] == 'ANCHOR'].loc[0]
        self.tbl_size = tbl_size
        self.img_size = img_size
        self.results = None
        self.reset()
        
    #-----------------------------------------------------------------------------
    def append_results(self, df):
        self.results = df if self.results is None else self.results.append(df)

    #-----------------------------------------------------------------------------
    def save_results(self, fname, ftype):
        if ftype == 'values':
            df = self.results.pivot(index='number', columns='field', values='value')       
            df = df[self.data.field] # sort columns
            df.to_excel(fname)
        else:
            self.results.to_excel(fname)
  
    #-----------------------------------------------------------------------------
    def reset(self):
        self.bbox = [0,0,0,0]
            
        df = self.data
        df.loc[:,"quality"] = 0
        df.loc[:,"text"]    = ''
        df.loc[:,"value"]   = ''
        for irow, row in df.iterrows():
            df.at[irow, 'bbox2'] = [0,0,0,0]
   
    #-----------------------------------------------------------------------------
    def locate_anchor_textbox(self, tboxes, min_row: int, min_match: float = 0.8):
        '''
        Locate anchor text box, based on text similarity and horizontal location

        Parameters
        ----------
        tboxes : array of dicts - page text-boxes. dicts have fields: "bbox" 
                for bounding box and "text" tor text
        min_row : int - only scan for text boxes below this row
        min_match: float - minimum text similarity level (fraction)

        Returns
        -------
        anchor_bbox: array of 4 integers - anchor text box location: (left, top, right bottom)

        '''
        x1, x2 = self.anchor.bbox[0], self.anchor.bbox[2]
        anchor_text = self.anchor.content.split('/')
        for tbox in tboxes:
            l, t, r, b = tbox['bbox']
            x   = (l+r) // 2
            y   = (t+b) // 2
            if x<x1 or x>x2 or y<min_row:
                continue
            txt = tbox['text'].strip()
            match = ppu.fuzzy_match(txt, anchor_text)
            if match>min_match:
                self.anchor.bbox2 = tbox['bbox']
                return tbox['bbox']
        return None

    #-----------------------------------------------------------------------------
    def locate_anchor_cell(self, cv_img, anchor_bbox, thresh = [0.9, 1.5]):
        
        cell_bbox = ppu.get_cell_borders(cv_img, anchor_bbox, thresh)
        if cell_bbox is None:
            cell_bbox = ppu.get_cell_fflood(cv_img, anchor_bbox, thresh)
        return cell_bbox

    #-----------------------------------------------------------------------------
    def add_cell(self, row, col, text, bbox):
        df = self.data
        index = np.where(df.location == (row, col))[0] # row number that meets the constraint   
        if len(index) == 0:
            return
        assert len(index) == 1, 'Cell not found'       # there can be only one
        idx = df.index[index[0]]                       # get row index 
        cell = df.loc[idx]  # get a row coresponding to this form cell       
            
        # remove static text present in most cells
        text  = text[1 if cell.type=='check' else 0]
        value = text
        quality = 0
        if isinstance(cell.content, str):

            # fuzze string matching
            match = 0
            for trg in cell.content.split('/'): # loop over possible patterns
                n = len(trg)
                src = text[:n]
                match = max(match,  SM(None, src.lower(), trg.lower()).ratio())
            
            if match>0.8:
                value = text[n:].strip() 
                quality += 1
        
        # process different cell types
        if cell.type=='num' and value.isnumeric():
            value = int(value)
            quality += 2
        elif cell.type=='check':
            if "["  in value: 
                quality += 2
            value = value.replace('[','').replace(']','').replace('J','').strip()
            value = (value != '')
        elif cell.type=='str':
            v = [p in value for p in punctuation]
            n = 1 - sum(v)/len(v) # fraction of characters which are not punctuation
            if n>0.7:
              quality += 2 # allow up to 30% punctuation  
            
        df.at[idx, 'quality'] = quality 
        df.at[idx, 'text']    = text 
        df.at[idx, 'value']   = value
        df.at[idx, 'bbox2']   = bbox

    #-----------------------------------------------------------------------------
    def scan_1(self, cv_img, tboxes, origin, draw_img=None):
        '''
        Based on image "img" and array of text boxes "tboxes" group text boxes 
        to fill a predefined form. Algorithm uses simple approach to find 
        horizontal and vertical lines that define the cells of the form.

        Parameters
        ----------
        img : Pillow Image - image of the current page
        tboxes : array of dicts - page text-boxes. dicts have fields: "bbox" 
                for bounding box and "text" tor text
        anchor_bbox: array of 4 integers - anchor text box location: (left, top, right bottom)

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        draw = ImageDraw.Draw(draw_img)
        font = ImageFont.truetype("times.ttf", size=40)
        origin = np.array(origin)
        
        l0, t0, r0, b0 = origin
        min_row = (t0+b0)//2
        hbars = ppu.find_lines(cv_img, 1)
        if hbars[-1]<b0:
            return 0
    
        # find index of the horizontal bar below the anchor (min_row)
        line_offset = max(0, np.argmax(hbars>min_row) )

        for iline in range(self.tbl_size[0]):
            if len(hbars)<=line_offset + iline+1:
                print('skip line', iline, len(hbars), line_offset + iline+1)
                break
            y1 = hbars[line_offset + iline]
            y2 = hbars[line_offset + iline+1]
            #line_img = cv_img.crop((0, y1, cv_img.shape[0], y2))
            line_img = cv_img[y1:y2, :]
            vbars = ppu.find_lines(line_img, 0)
            nbox = min(self.tbl_size[1], len(vbars)-1)
            for ibox in range(nbox):
                x1 = vbars[ibox]
                x2 = vbars[ibox+1]
                bbox = [x1, y1, x2, y2]

                text1 = []
                text2 = []
                for tbox in tboxes:
                    if bbox_inside(tbox['bbox'], bbox, True):
                        text1.append( tbox['text'])                    
                    if bbox_inside(tbox['bbox'], bbox, False):
                        text2.append( tbox['text'])
                if len(text2)>0:
                    bbox0 = bbox_offset(bbox, -origin)
                    text = [' '.join(text1), ' '.join(text2)]
                    self.add_cell(iline, ibox, text, bbox0 )
                    # draw.rectangle(bbox, fill =None, outline ="cyan", width=3)
                    draw.text(bbox[:2], ' '.join(text2), fill="red", font=font) 
                    
        self.bbox = (0, hbars[line_offset], cv_img.shape[0], y2)   
        return self.data['quality'].sum()
    
    #-----------------------------------------------------------------------------
    def scan_2(self, cv_img, tboxes, origin, draw_img=None):
        '''
        Based on image "img" and array of text boxes "tboxes" group text boxes 
        to fill a predefined form.

        Parameters
        ----------
        img : Pillow Image - image of the current page
        tboxes : array of dicts - page text-boxes. dicts have fields: "bbox" 
                for bounding box and "text" tor text
        anchor_bbox: array of 4 integers - anchor text box location: (left, top, right bottom)


        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        if draw_img is not None:
            draw = ImageDraw.Draw(draw_img)
            font = ImageFont.truetype("times.ttf", size=40)
            
        max_y = 0
        for irow, row in self.data.iterrows():
            fname1 = None #fname.replace('.png', '_{:02d}.png'.format(irow))
            iline, ibox = row['location']
            bbox0 = bbox_offset(row['bbox'], origin)
            bbox1 = ppu.get_cell_fflood(cv_img, bbox0, debug_fname=fname1)
            draw.rectangle(bbox0, fill =None, outline ="cyan", width=3)
            #draw.text(bbox0[:2], row.field, fill="red", font=font) 
            if bbox1 is None:
                print('    cell flood fill failed:', iline, ibox, origin[0], origin[1])
                continue
            draw.rectangle(bbox1, fill =None, outline ="green", width=3)
            text1 = []
            text2 = []
            for tbox in tboxes:
                if bbox_inside(tbox['bbox'], bbox1, True):
                    text1.append( tbox['text']) # text fully inside
                if bbox_inside(tbox['bbox'], bbox1, False):
                    text2.append( tbox['text']) # text center inside           
            if len(text2)>0:
                bbox2 = bbox_offset(bbox1, -origin)
                max_y = max(max_y, bbox2[3])
                text = [' '.join(text1), ' '.join(text2)]
                self.add_cell(iline, ibox, text, bbox2)
                draw.text(bbox1[:2], ' '.join(text2), fill="red", font=font) 

            #print('    Cell found:', ' '.join(text))

        self.bbox = (0, origin[1], cv_img.shape[0], max_y) 

        return self.data['quality'].sum()            
            
    # ---------------------------------------------------------------------------
    def scan_page(self, ipage, ibox, pil_img, tboxes, min_quality, debug_fname=None):
    
        draw_img = pil_img.copy()
        cv_img   = np.array(pil_img) 
        font = ImageFont.truetype("times.ttf", size=40)
        draw = ImageDraw.Draw(draw_img)
        
        # convert image from color to boolean
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cv_img[cv_img> 127] = 255 # make into boolean
        cv_img[cv_img<=127] = 0 # make into boolean
    
        min_row = 0
        while True:
            self.reset()
            
            # identify anchor text-box
            anchor_bbox = self.locate_anchor_textbox( tboxes, min_row)
            if anchor_bbox is None:
                break
            anchor_bbox[2] += int(0.9*self.img_size[0])
            min_row = anchor_bbox[3]
            
            # locate cell on the form that contain the anchor text-box
            anchor_cell = self.locate_anchor_cell(cv_img, anchor_bbox)
            draw.rectangle(anchor_bbox, fill=None, outline ="magenta", width=3)
            if anchor_cell is None:
                print('   anchor cell flood fill failed')
                continue
            else:
                draw.rectangle(anchor_cell, fill=None, outline ="red", width=3)
                draw.text(anchor_cell[:2], 'anchor', fill="red", font=font) 
            anchor_cell = np.array(anchor_cell)

    
            alg = 1
            quality = self.scan_1(cv_img, tboxes, anchor_cell, draw_img=draw_img)
    
            if quality<min_quality:
                alg = 2
                quality = self.scan_2(cv_img, tboxes, anchor_cell, draw_img=draw_img)
                
            print(ipage, ibox, quality, alg)
            if quality>min_quality:
                df = self.data.copy()
                df['page']   = ipage
                df['number'] = ibox
                df['overall_quality'] = quality
                self.append_results(df)
            ibox += 1
        
        if debug_fname:
            draw_img.save(debug_fname)
        
        return ibox



