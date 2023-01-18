# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 07:21:19 2022

@author: tuszynskij
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 2022

@author: Jarek Tuszynski
"""
# importing external modules
import os
import cv2
import math
import numpy as np

from matplotlib import pyplot as plt

# importing local modules
import pdf_parser_utils as ppu

# ---------------------------------------------------------------------------
def cyclic_intersection_pts(pts):
    """
    Sorts 4 points in clockwise direction with the first point been closest to 0,0
    Assumption:
        There are exactly 4 points in the input and
        from a rectangle which is not very distorted
    """
    if pts.shape[0] != 4:
        return None

    # Calculate the center
    center = np.mean(pts, axis=0)

    # Sort the points in clockwise
    cyclic_pts = [
        # Top-left
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] < center[1]))[0][0], :],
        # Top-right
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] < center[1]))[0][0], :],
        # Bottom-Right
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] > center[1]))[0][0], :],
        # Bottom-Left
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] > center[1]))[0][0], :]
    ]

    return np.array(cyclic_pts)

# ---------------------------------------------------------------------------
def rotate(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    angle *= 180/math.pi
    ox, oy = origin
    px, py = points[:,0], points[:,1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.column_stack((qx, qy)).astype(int)
 
# ---------------------------------------------------------------------------
def main():
    # extract image pages
    folder_in  = r".\input"
    fname_in  = os.path.join(folder_in,  '56092341 EMEX PM02 - eDCRO_337684.pdf')  
    img = ppu.extract_pdf_page(fname_in, 70, None)
    mat1 = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    
    pts1 = np.array([[84,443],[2457,3256],[2424,355],[109,3250]])
    pts2 = cyclic_intersection_pts(pts1)
    
    x,y,w,h = cv2.boundingRect(pts2)
    pts3 = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        
    # Get the transform
    m = cv2.getPerspectiveTransform(np.float32(pts2), np.float32(pts3))
    # Transform the image
    mat2 = cv2.warpPerspective(mat1, m, np.flip(mat1.shape) )
    
    out = np.array(img)
    cv2.drawContours(out, [pts2], -1, (255, 0, 0), 10) # blue
    cv2.drawContours(out, [pts3], -1, (0, 255, 0), 10) # green
    # cv2.drawContours(out, [pts4, pts5], -1, (0, 0, 255), 10) # red
    
    cv2.imwrite("deskew1.jpg", out)   
    cv2.imwrite("deskew2.jpg", mat2)


# =============================================================================
if __name__ == "__main__":
    main()