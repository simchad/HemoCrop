"""Module for crop hemocytometer grid (1mm^2)

This module provides recognize and crop the grid of hemocytometer,
1mm x 1mm square grid, mainly use cv2.HoughLines() function.

Available functions:
- check_type_image: Check type if input image is np.ndarray.
- check_type_df: Check type if input df is pd.Dataframe.
- show_image: Show image using with matplotlib.pyplot module.
- roundup_img_wh: Rounds up width and height of image to the hundredths place.
- img_rotate_size: Modified width and height size of rotated image.
- nth_gaussian_blur: The gaussian_filter with N-th repetitions
- pre_crop: Crop the image in advance (Still build-up function).
- houghline_coord: Generate the lines using with cv2.HougLine module.
- AlignIimage: Align the image based on the chamber lattice.
- CropImage: Cut-off the outside of chamber lattice.

USER PATHS:
- HOMEDIR: (default)
- HOMEDIR+img_jpg: PATH for input image
- HOMEDIR+img_output: PATH for output image

Updates:
2023-04-17
- Docstring, Log file improves.
- Function: AlignImage (img_aligned[hh:h-hh, ww:w-ww]) error.
2024-07-12
- Every hemo images freely use.
"""
__version__ = '1.0.0'
__author__ = 'Hyunchae Sim'

import cv2
import math
import numpy as np
import os
import pandas as pd
import re
import statistics as stats
import matplotlib.pyplot as plt


# Get current login user name & change directory to home.
#HOMEDIR = "C:/Users/"+os.getlogin()+"/Documents/autocellcounter/"
HOMEDIR = "C:/Users/"+os.getlogin()+"/Documents/GitHub/HemoCrop/"
os.chdir(HOMEDIR)

# -----------------------------------------------------------------------
# Function : Check instances
def check_type_image(img):
    """check_type_image(img) -> TypeError"""
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be a NumPy array.")


def check_type_df(df):
    """check_type_df(df) -> TypeError"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
# -----------------------------------------------------------------------
# Function: Side-stream functions
def show_image(img):
    """show_image(img) -> plt.show()

    Parameter
    ---------
    * img (numpy.ndarray) : Input must be rgb or gray numpy.ndarray type.
    """
    check_type_image(img)
    if len(img.shape) == 2:
        # Gray-scale
        plt.imshow(img)
        plt.show()
    elif len(img.shape) == 3:
        # RGB-scale
        b, g, r = cv2.split(img)
        plt.imshow(cv2.merge([r, g, b]))
        plt.show()
    else:
        # Unknown
        raise TypeError("Input image may not rgb or gray image.")


def roundup_img_wh(img):
    """
    roundup_img_wh(img) -> h, w (integer)

    Parameter
    ---------
    * img (numpy.ndarray) : Input must be a numpy.ndarray type.
    """
    check_type_image(img)
    height, width = img.shape[:2]
    # math.ceil Ex) 768 -> 7.68 -> 8.00 -> 800
    h = math.ceil(height/100)*100
    w = math.ceil(width/100)*100
    return (h, w)


def img_rotate_size(image, theta: float = 0):
    """
    size_rotate_img(image, theta: float=0) -> tuple(ww, hh)

    Parameter
    ---------
    * image (numpy.ndarray) : Input must be a numpy.ndarray type.
    * theta (float) : Normally use median value of houghline transformed degree.

    Concept
    -------
    If you rotate a square-shaped image, the width and height of the rotated image will be larger than the original.
    The image will still be scaled, but with a larger coordinate range, creating a margin.
    The colour difference between the background and the original image caused by this margin creates a noise line
    during the hough-line conversion, causing errors in the subsequent process.
    To solve this, we find the size of the margin and crop the edge by the margin after the alignment process to avoid noise.

    Notes
    -----
    From a rectangle rotated by theta, we can use trigonometry to find the size of the rotated image.
    ww: w*cos(theta)+h*sin(theta)
    hh: h*cos(theta)+w*sin(theta)
    Width margins: long-side <- w*cos(theta), short-side <- h*sin(theta)
    Height margins: long-side <- h*cos(theta), short-side <- w*sin(theta)
    """
    check_type_image(image)
    h, w = image.shape[:2]
    rads = math.radians(theta)
    ww = w*math.cos(rads)+h*math.sin(rads)
    hh = h*math.cos(rads)+w*math.sin(rads)
    return (ww, hh)


# Function : Main-stream functions
def nth_gaussian_blur(img, kernel_size: int = 9, repetition: int = 1):
    """
    nth_gaussian_blur(img, kernel_size, repetition) -> img

    Parameter
    ---------
    * img (numpy.ndarray) : Input image must be numpy.ndarray type.
    * kernel_size (int, odd) : Input must be odd-integer.
    * repetition (int) : The number of times that repeat gaussian-reducing.
    
    Return
    ------
    numpy.ndarray object (img)
        The object which was gaussian-blurred image.
    """
    # Validation
    check_type_image(img)
    if kernel_size%2 == 0:
        raise ValueError("(parameter) kernel_size must be a odd number.")
    # Process
    for i in range(repetition):
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img


def pre_crop(img, list1: list, list2: list):
    """
    pre_crop(img, list1, list2) -> val
    (Caution to use) : Still build up.

    Parameter
    ---------
    * img (numpy.ndarray): Input image must be type.
    * list1 (list) : 
    * list2 (list) :
    """
    check_type_image(img)
    img_shape = img.shape[:2]
    val = [min(min(list1), min(list2)), max(max(list1), max(list2))]

    crop_min = math.floor(val[0]/100)*100
    crop_max = math.ceil(val[1]/100)*100

    # Cases. The coord is out of range OR Much empty space
    # coord <= 0 -> set 0 OR coord >= w, h -> set w, h
    # min side
    if crop_min <= 0:
        val[0] = 0
    else:
        val[0] = math.ceil(crop_min/200)*100
    # max side
    if crop_max >= img_shape:
        val[1] = img_shape
    else:
        val[1] = math.ceil((crop_max + img_shape)/200)*100
    return val


def houghline_coord(img_rgb, img_edge, rho: int = 1, theta: float = 0.5, threshold: int = 190):
    """
    houghline_coord(img_rgb, img_edge, rho, theta, threshold) -> coordinate, img_hlined

    Parameter
    ---------
    * img_rgb (numpy.ndarray) : Image must be rgb-colored image.
    * img_edge (numpy.ndarray): Image mus be gray-scale image.
    * rho (int) : Param from cv2.HoughLines.
    * theta (float) : Concepts from cv2.HoughLines, step-by-step increasing degree value of line.
    Rough(1), Quick(0.5), Precise(0.1, DEMO)
    * threshold (int) : See cv2.HoughLines.

    Return
    ------
    pandas.DataFrame object (coordinate)
        The object which containing coordinates of HoughLines (x1, y1, x2, y2, deg).
    numpy.ndarray object (img_hlined)
        The object which overlayed with HoughLines on img_rgb.copy().

    Notes
    -----
    This function create pd.DataFrame that contain coordinate of lines generated from
    cv2.HoughLine(). The second return value is numpy.ndarray type image should be use
    validation. The intercept of Houghline transformed line is set to round-up value of
    height and width of input image shape. After image alignment, you can set theta to 90
    inducing HoughLine transform only horizontal and vertical lines.
    """
    if not isinstance(img_rgb, np.ndarray) or not isinstance(img_edge, np.ndarray):
        raise TypeError("Input must be a numPy.ndarray.")
    img_tmp = img_rgb.copy()

    # define return formats
    coord_list = []
    # coordinate = pd.DataFrame(columns=('x1', 'y1', 'x2', 'y2', 'deg'))

    # parameter.img
    height, width = img_edge.shape[:2]
    h = math.ceil(height/100)*100
    w = math.ceil(width/100)*100

    # parameter.numeric.theta indicate stepped increase degree of line., 1 degree = pi/180 radian
    theta = np.pi/(180/theta)
    lines = cv2.HoughLines(img_edge, rho, theta, threshold)
    
    for i in range(len(lines)): #Enumerate
        for rho, theta in lines[i]:
            a = np.cos(theta) # np.cos(90) = 0 --> Horizontal Line
            b = np.sin(theta) # np.sin(90) = 1 --> Vertical Line
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + w*(-b))
            y1 = int(y0 + h*(a))
            x2 = int(x0 - w*(-b))
            y2 = int(y0 - h*(a))
            img_hlined = cv2.line(img_tmp,(x1,y1),(x2,y2),(0,0,255),1)
            rad = math.atan2((y2-y1), (x2-x1))
            deg = math.degrees(rad)
            coord_list.append([x1, y1, x2, y2, deg])
    coordinate = pd.DataFrame(coord_list, columns=['x1', 'y1', 'x2', 'y2', 'deg'])

    # Validation: coordinate.csv and img.jpg
    # coordinate.to_csv(path_or_buf=HOMEDIR+'coord/coord.csv', encoding='utf-8')
    return coordinate, img_hlined


def AlignImage(img, df):
    """
    AlignImage(img, df) -> img_aligned (img)

    Parameter
    ---------
    * img (numpy.ndarray) : Image must be numpy.ndarray type.
    * df (Pandas.DataFrame) : Dataframe that contain coordinate of all houghlines.

    Return
    -----
    numpy.ndarray object (img_alined)
        The object which aligned by median value of degree within coordinate df.

    Notes
    -----
    Rotate image around by centerpoint with median value of degree list
    """
    # Validation
    check_type_image(img)
    check_type_df(df)

    # Here
    h, w =img.shape[:2]
    list_deg = df['deg'].to_list()
    for i, deg in enumerate(list_deg):
        if deg < 0:
            list_deg[i] += 90
    deg_median = stats.median(list_deg)
    rads = math.radians(deg_median)
    ww = 2*math.ceil(w*math.sin(rads))
    hh = 2*math.ceil(h*math.sin(rads))
    cp = (w/2, h/2)
    rot = cv2.getRotationMatrix2D(cp, deg_median, 1)
    img_aligned = cv2.warpAffine(img, rot, (w, h))
    # img_aligned = img_aligned[hh:h-hh,ww:w-ww]
    # -> Error
    return img_aligned, hh, ww


def CropImage(img=None, coordinates=None):
    """
    CropImage(img, coordinate) -> img_grid

    Parameter
    ---------
    @img (numpy.ndarray) : Input image must be numpy.ndarray type.
    Set img to the current img to crop.
    @coordinate (pandas.DataFrame) : Input must be pandas.dataframe type.
    Mostly, coordinate referred from houghline_coord[0].

    Return
    -----
    numpy.ndarray object (img_grid)
        The object which cropped around the grid of image.

    Notes
    -----
    Typically 100x zoom-in hemocytometer, width of line is 10~20, 0.25x0.25mm^2 area is 70~80px.
    Every lines have coordinate. The betweeness of lines are induced from differential.
    Case of Horizontal line, lines have -width~+width coords and variable y coorinate,
    vertical line is against to horizontal case.
    Of course, the line within grid has thickness. Meaning that, a line has two hough line coordinate.
    For this circumstance, the cell could overapped on the line
    Considering thickness of grid-line, Both-side end-value needed.
    """
    # Validation   
    check_type_image(img)
    check_type_df(coordinates)

    size_v, size_h = roundup_img_wh(img)

    # Part 1. Separation vertical and horizontal lines
    coords = round(abs(coordinates.copy(deep=True)), -1)
    is_not_hor = coords.loc[coords['x1'] != size_h].index
    is_not_ver = coords.loc[coords['y1'] != size_v].index

    coord_hor = coordinates.drop(index=is_not_hor)
    coord_ver = coordinates.drop(index=is_not_ver)

    # Drop columns and diiferential
    coord_hor = coord_hor.drop(['x1', 'x2', 'deg'], axis=1).sort_values('y1')
    coord_hor['y1_diff'] = coord_hor['y1'].diff()
    coord_hor.reset_index(drop=True, inplace=True)

    coord_ver = coord_ver.drop(['y1', 'y2', 'deg'], axis = 1).sort_values('x1')
    coord_ver['x1_diff'] = coord_ver['x1'].diff()
    coord_ver.reset_index(drop=True, inplace=True)

    # Valid size for 1mm x 1mm grids (4*4 grid)
    index_y = coord_hor[(coord_hor['y1_diff'] > 75) & (coord_hor['y1_diff'] <= 85)].index
    index_x = coord_ver[(coord_ver['x1_diff'] > 75) & (coord_ver['x1_diff'] <= 85)].index

    # Part 2. Precise crop
    grid_line = {"x1":0,
                 "x2":0,
                 "y1":0,
                 "y2":0
                 }

    # Grid Line -> |, Grid Lattice -> ㅁ
    # ||ㅁ||ㅁ||ㅁ||ㅁ|| shape.
    # 
    # Grid line limiting sequence
    #
    # ****
    #
    # Parmas
    # ------
    # index_xy = index_y or index_x
    # coord_axis = coord_hor or coord_ver
    #
    # Returns
    # -------
    # int (xy1, xy2) --> dict grid_line{x1,x2,y1,y2}
    # if min(index_xy) >= 2:
    #     grid_xy_min = coord_axis.iloc[min(index_xy)-2]
    #     grid_line["y1"] = int(max(grid_xy_min[0], grid_xy_min[1]))
    # else:
    #     grid_xy_min = coord_axis.iloc[0]
    #     grid_line["y1"] = int(max(grid_xy_min[0], grid_xy_min[1]))
    # if max(index_xy) == coord_axis.shape[0]-1:
    #     grid_xy_max = coord_axis.iloc[max(index_xy)]
    # else:
    #     grid_xy_max = coord_axis.iloc[max(index_xy)+1]
    # grid_line["y2"] = int(min(grid_xy_max[0], grid_xy_max[1]))
    #
    # ****
    
    # For Horizontal Lines
    # try - except - finally 구문으로 succes, fail 구현
    if min(index_y) >= 2:
        grid_y1 = coord_hor.iloc[min(index_y)-2]
        grid_line["y1"] = int(max(grid_y1[0], grid_y1[1]))
    else:
        grid_y1 = coord_hor.iloc[0]
        grid_line["y1"] = int(max(grid_y1[0], grid_y1[1]))
    if max(index_y) == coord_hor.shape[0]-1:
        grid_y2 = coord_hor.iloc[max(index_y)]
    else:
        grid_y2 = coord_hor.iloc[max(index_y)+1]
    grid_line["y2"] = int(min(grid_y2[0], grid_y2[1]))

    # For Vertical Lines
    if min(index_x) >= 2:
        grid_x1 = coord_ver.iloc[min(index_x)-2]
        grid_line["x1"] = int(max(grid_x1[0], grid_x1[1]))
    else:
        grid_x1 = coord_ver.iloc[0]
        grid_line["x1"] = int(max(grid_x1[0], grid_x1[1]))
    if max(index_x) == coord_ver.shape[0]-1:
        grid_x2 = coord_ver.iloc[max(index_x)]
    else:
        grid_x2 = coord_ver.iloc[max(index_x)+1]
    grid_line["x2"] = int(min(grid_x2[0], grid_x2[1]))
    
    # Confirm size of image 340x340
    gap_x = abs(grid_line["x2"] - grid_line["x1"])
    gap_y = abs(grid_line["y2"] - grid_line["y1"])
    if gap_x != 340:
        grid_line["x2"] = grid_line["x2"] + (340 - gap_x)
    if gap_y != 340:
        grid_line["y2"] = grid_line["y2"] + (340 - gap_y)

    # 2023.04.01. 11:48 AM
    # x2, y2 쪽으로 더 많이 잘리는 문제: 오히려 x1, y1 쪽으로 margin 늘려야함.
    # x1, y1이 제대로 outline이 걸리지 않고 outline의 inner line이 걸리면, 그 gap 만큼 x2, y2 방향으로 쏠리게됨.
    # x1, y1 방향에서 xy.diff 확인해서 gird_line 두께 (3~4 px) 있는지 확인해야.
    # IF, 3~4 px 정도 grid_line 없다면, x1, y1 방향으로 늘려야 해.

    # IF x2, y2 always bigger than x1, y1
    img_grid = img[grid_line["y1"]:grid_line["y2"], grid_line["x1"]:grid_line["x2"]]
    # cv2.imwrite(HOMEDIR+'img_output/img_crop.jpg', img_grid)
    return img_grid, coord_hor, coord_ver


if __name__ == "__main__":
    CWD = os.getcwd()
    PATH = os.path.join(CWD, 'img_jpg')
    path_valid = os.path.join(CWD, 'validation/process_log.csv')
    file_list = os.listdir(PATH)
    file_log = []
    rst_success = 0
    rst_failed = 0

    for i, file in enumerate(file_list):
        # enumerate exception how ignore?
        # Few images can not work: file_list[:20]
        img = cv2.imread(os.path.join(PATH,file), cv2.IMREAD_COLOR)
        rst = "Grid_"+str(i)+"_%s.jpg"%re.sub(r"\.[a-z]*","",file)
        PATH_GRID =HOMEDIR+"img_output/"+rst

        try:
            # (1) Canny-Align
            img_L1 = nth_gaussian_blur(img, 9, 2)
            img_L2 = cv2.cvtColor(img_L1, cv2.COLOR_RGB2GRAY)
            img_edge = cv2.Canny(img_L2, 10, 20)
            img_hough = houghline_coord(img, img_edge, theta=0.5)
            img_current = AlignImage(img, img_hough[0])

            # (2) Grid crop
            img_L1 = nth_gaussian_blur(img_current[0], 7, 1) #7
            img_L2 = cv2.cvtColor(img_L1, cv2.COLOR_RGB2GRAY)
            img_edge = cv2.Canny(img_L2, 5, 10)
            img_tmp = houghline_coord(img_current[0], img_edge, theta=90)
            img_grid = CropImage(img_current[0], img_tmp[0])

            file_log.append([file, "Success", rst])
            cv2.imwrite(PATH_GRID, img_grid[0])
            rst_success += 1
        except:
            file_log.append([file, "Failed", ""])
            rst_failed += 1

    file_log = pd.DataFrame(file_log, columns=['Input image', 'Proceed', 'Output image'])
    file_log.to_csv(path_or_buf=path_valid, sep=',', encoding='utf-8')

    with open("process_log.txt", "w") as output:
        output.write(str(file_log))

    # Report the process result.
    print("Report: In total %d files processed, %d files succeed, and %d files truncated...!"
          %(len(file_list), rst_success, rst_failed))

    
    # Upcomming ---
    # issue: Grid outline does not detect
    # -> solution: grid line 앞뒤로 line 들어있는지 확인하면 됨. pixel이 4짜리가 들어있는지.
    # -> 들어있지않다면, inline으로 판단하고 +-4px.
    # 2023-04-15: high-resolution 작동안하는 문제.
    #
    # 2025-08-29
    # HOMEDIR have to change join though.
    # 주기능 정상 작동. 
    # Report: In total 206 files processed, 180 files succeed, and 26 files truncated...! error rate (12.6%)
    # Process 180개 중, 7개 심각한 문제 3.8% (7/180)