# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 2022

@author: Jarek Tuszynski
"""
# importing external modules
import os
import fitz  # load PyMuPDF
import numpy as np
import pandas as pd

# importing local modules
import pdf_parser_utils as ppu
import pdf_parser_forms as ppf


# ---------------------------------------------------------------------------
def scan_PMO_for_object_sections(file_in, form_df, path_out):
    tt   = ppf.form_parser(form_df)
    doc  = fitz.open(file_in) # now a fitz.Document class
    ibox = 0
    min_quality = 50
    for ipage in range(5,69): # 69
        page    = doc[ipage] # now a fitz.Page class
        pil_img = ppu.pixelmap2pil(ppu.extract_pdf_image(page, False))
        tboxes  = ppu.extract_pdf_text (page, True)
        fname_out = os.path.join('./output', 'page_{:03d}.png'.format(ipage))
        ibox = tt.scan_page(ipage, ibox, pil_img, tboxes, min_quality, fname_out)

    tt.save_results('PMO_objects_details.xlsx', 'details')
    tt.save_results('PMO_objects_values.xlsx', 'values')

# ---------------------------------------------------------------------------
def scan_all_files(folder_in, folder_out, overlay_text):
    for fname in os.listdir(folder_in):
        fname_in = os.path.join(folder_in, fname)
        if os.path.isfile(fname_in) and fname_in.endswith('.pdf'):
            print(fname_in)
            froot = fname.split('.')[0] # strip the extenson
            fname_out = os.path.join(folder_out, froot)
            ppu.save_pdf_pages(fname_in, fname_out, overlay_text)

# ---------------------------------------------------------------------------
def main():
    # extract image pages
    folder_in  = r".\input"
    folder_out = r".\output"
    overlay_text = 'none'
    #scan_all_files(folder_in, folder_out, overlay_text)

    # extract images and text
    fname_in  = os.path.join(folder_in,  '56092341 EMEX PM02 - eDCRO_337684.pdf')
    fname_out = os.path.join(folder_out, '56092341 EMEX PM02 - eDCRO_337684')
    fname_in  = os.path.join(folder_in,  '56092562 EF-69 PM02 - eDCRO_337685.pdf')
    fname_out = os.path.join(folder_out, '56092562 EF-69 PM02 - eDCRO_337685')
    overlay_text = 'flat'
    #overlay_text = 'none'
    ppu.save_pdf_pages(fname_in, fname_out, overlay_text)
    return

    folder_out = r".\output_obj"
    form_df = pd.read_csv('Form layout (PMO object).csv')
    scan_PMO_for_object_sections(fname_in, form_df, folder_out)
    return

    fname0_out = './56092341 EMEX PM02 - eDCRO_337684_page_000.png'
    fname5_out = './56092341 EMEX PM02 - eDCRO_337684_page_005.png'
    #ppu.extract_pdf_page(fname_in, 0).save(fname0_out)
    ppu.extract_pdf_page(fname_in, 5).save(fname5_out)

    # for i_img, img in enumerate(ppu.extract_raw_pdf_images(fname_in, 0)):
    #     fname0_out = './56092341 EMEX PM02 - eDCRO_337684_page_000_{}.png'.format(i_img)
    #     img.save(fname0_out)

# =============================================================================
if __name__ == "__main__":
    main()