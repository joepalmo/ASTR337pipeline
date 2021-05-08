'''
ASTR337 DATA REDUCTION MODULE 2020

AUTHOR: JOE PALMO
'''

# The standard fare:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.visualization import ZScaleInterval
# Recall our use of this module to work with FITS files in Lab 4:
from astropy.io import fits
import os 
import glob 
import scipy.signal
import scipy.ndimage.interpolation as interp
import shutil
import pdb


def filesorter(filename, foldername, fitskeyword_to_check, keyword):
    '''
    Takes .fit files with a specific keyword value in their header and sends them to the specified
    folder-- if the folder does't exist, it creates it.
    
    Inputs:
    filename (str) - name of .fits file
    foldername (str) - name of folder that the .fits file may be sorted to
    fitskeyword_to_check (str) - if the value of the header category keyword matches this input, then the .fits file will 
                                    be sorted to foldername
    keyword (str) - the header category that is being checked
    '''
    # Checking to see if the filename exists, if it doesn't, it prints that the filename does not exist or that it has been moved.  
    if os.path.exists(filename):
        pass
    else:
        print(filename + " does not exist or has already been moved.")
        return
    #acquiring all 'column' headers for specified filename, assigning it to header
    #then assigning values under specified header keyword to fits_type
    header = fits.getheader(filename)
    fits_type = header[keyword]
    
    # Checking to see if foldername exists. If it does, move on, if it doesn't, it generates a folder of that name.
    if os.path.exists(foldername):
        pass
    else:
        print("Making new directory: " + foldername)
        os.mkdir(foldername)
    
    # Checks to see if the fits_type we got from the keyword header matches the fitskeyword we are looking for. If it does, it moves it into the destination which is specified by the
    # foldername we inputted into the function. If it does not, it just doesn't do anything. 
    if fits_type == fitskeyword_to_check:
        destination = foldername + '/'
        print("Moving " + filename + " to: ./" + destination + filename)
        os.rename(filename, destination + filename)  
    return

def sortall(master_directory):
    '''
    First sorts all the files into four different folders based on their IMGTYPE: flats, biasframes, darks, and lights.
    Then, the darks are sorted by EXPTIME, and the flats are sorted by FILTER.
    
    Inputs:
    master_directory (str) - the directory where all of the raw data lives
    '''
    #change directory to input
    os.chdir(master_directory)
    #glob all .fit files
    all_fits = glob.glob("*.fit")
    #sort files into the four folders
    for fitsfile in all_fits:
        filesorter(fitsfile, 'flats', 'Flat Field', 'IMAGETYP')
    for fitsfile in all_fits:
        filesorter(fitsfile, 'biasframes', 'Bias Frame', 'IMAGETYP')
    for fitsfile in all_fits:
        filesorter(fitsfile, 'darks', 'Dark Frame', 'IMAGETYP')        
    for fitsfile in all_fits:
        filesorter(fitsfile, 'lights', 'Light Frame', 'IMAGETYP')
        
    #sort the darks by exposure time
    os.chdir(master_directory+'/darks/')
    dark_fits = glob.glob('*.fit')
    times = []
    for i in dark_fits:
        headers = fits.getheader(i)
        times.append(headers['EXPTIME'])
        print(headers['EXPTIME'])
    times = np.array(times)
    uniquetimes = np.unique(times)
    for i in range(len(uniquetimes)):
        for yeet in dark_fits:
            filesorter(yeet, str(uniquetimes[i]) + 'sec', uniquetimes[i], 'EXPTIME')
            
    #sort the flats by filter
    os.chdir(master_directory+'/flats/')
    flat_fits = glob.glob('*.fit')
    for filename in flat_fits:
        filesorter(filename, 'Blue', 'Blue', 'FILTER')
    for filename in flat_fits:
        filesorter(filename, 'Visual', 'Visual', 'FILTER')
    for filename in flat_fits:
        filesorter(filename, 'Red', 'Red', 'FILTER')  
    return

def mediancombinenan(filelist):
    '''
    The function takes 1 input, a list of fits files to be combined, and takes the median
    pixel values for each pixel, outputting a median fits file. This function ignores naN values.
    
    Inputs:
    filelist (array-like) - a list of strings that are filepaths to images to be combined
    '''
    # Defines variable holding the length of the inputted list of files.
    n = len(filelist)
    
    # Assigns a name to the array data of the first file in the list.
    first_frame_data = fits.getdata(filelist[0])
    
        # Puts the y and x dimensions of the image into variables.
    imsize_y, imsize_x = first_frame_data.shape
    
    # Creates an empty three-dimensional array which will store the data from the entire list of files.
    fits_stack = np.zeros((imsize_y, imsize_x , n)) 
    
    # Goes through the list of files and sets the corresponding element in the array to the value of the
    # current pixel.
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        hed = fits.getheader(filelist[ii])
        if (hed['EXPTIME']> 60):
            im_normal = im/hed['EXPTIME']
            im_final = im_normal*60
        else: im_final = im
        fits_stack[:,:,ii] = im_final
        
    # Uses a numpy function to create a median frame along the third axis, which it then returns.        
    med_frame = np.nanmedian(fits_stack, axis = 2)
    
    return med_frame

def mediancombine(filelist):
    '''
    The function takes 1 input, a list of fits files to be combined, and takes the median
    pixel values for each pixel, outputting a median fits file.
    
    Inputs:
    filelist (array-like) - a list of strings that are filepaths to images to be combined
    '''
    # Defines variable holding the length of the inputted list of files.
    n = len(filelist)
    
    # Assigns a name to the array data of the first file in the list.
    first_frame_data = fits.getdata(filelist[0])
    
    # Puts the y and x dimensions of the image into variables.
    imsize_y, imsize_x = first_frame_data.shape
    
    # Creates an empty three-dimensional array which will store the data from the entire list of files.
    fits_stack = np.zeros((imsize_y, imsize_x , n)) 
    
    # Goes through the list of files and sets the corresponding element in the array to the value of the
    # current pixel.
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        fits_stack[:,:,ii] = im
        
    # Uses a numpy function to create a median frame along the third axis, which it then returns.        
    med_frame = np.nanmedian(fits_stack, axis = 2)
    
    return med_frame

def bias_subtract(filename, path_to_bias):
    '''
    This function takes two inputs: name of a file and a file path to the master bias and subtracts the master bias from the data
    It creates a file with the prefix 'b_', the new data, the header, and overwrites any existing files with the same name
    
    Inputs:
    filename (str) - filepath to image that is to be bias subtracted
    path_to_bias (str) - filepath to master bias image
    '''
    #grab information from input filepaths
    data = fits.getdata(filename)
    header = fits.getheader(filename)
    mb = fits.getdata(path_to_bias)
    
    #bias subtraction
    subtracted = data - mb

    #save the file
    fits.writeto('b_' + filename, subtracted, header, overwrite=True)
    return 

def getmasterbias(biasdirectory):
    '''
    This function creates a master bias frame given the path to the bias frames.
    
    Inputs:
    biasdirectory (str) - filepath to the biasframes folder
    
    Outputs:
    master_bias_path (str) - filepath to the master bias image
    '''
    
    os.chdir(biasdirectory)
    biasfiles = glob.glob('*.bias*.fit')
    #median combine all bias frames, and save the master
    fits.writeto('Master_Bias.fit', mediancombinenan(biasfiles), fits.getheader(biasfiles[0]), overwrite=True)
    master_bias_path = os.getcwd() + '/Master_Bias.fit'
    return(master_bias_path)
    
def getmasterdark(master_bias_path, darkdirectory):
    '''
    This function creates a master dark frame given the path to the dark frames.
    
    Inputs:
    master_bias_path (str) - filepath to the master bias image
    darkdirectory (str) - filepath to the darks folder
    
    Outputs:
    master_darkpath (str) - filepath to the master dark image
    '''
    
    os.chdir(darkdirectory)
    darkfiles = glob.glob('*.fit')
    #first must bias subtract all darks
    for i in darkfiles:
        bias_subtract(i, master_bias_path)
    biassubtracteddarks = glob.glob('b_*.fit')
    #combine to create a master dark
    CBSD = mediancombinenan(biassubtracteddarks)
    #save master dark
    fits.writeto('Master_Dark.fit', CBSD, fits.getheader(biassubtracteddarks[0]), overwrite = True)
    master_darkpath = os.getcwd() + '/Master_Dark.fit'
    return(master_darkpath)

def dark_subtract(filename, path_to_dark):
    '''
    Takes an image file name and a path to the master dark frame,
    normalizes the dark current and scales it to the image exposure time,
    subtracts it, and writes the resulting file to the working directory.
    
    Inputs:
    filename (str) - filepath an image
    path_to_dark (str) - filepath to the master dark image
    '''
    # Your code goes here.
    
    fileframe = fits.getdata(filename)
    fileheader = fits.getheader(filename)
    darkframe = fits.getdata(path_to_dark)
    darkheader = fits.getheader(path_to_dark)
    if (darkheader['EXPTIME'] != fileheader['EXPTIME']):
        masterdarknormalized = darkframe/darkheader['EXPTIME']
        exposure = fileheader['EXPTIME']
        darktouse = masterdarknormalized*exposure
    else:
        darktouse = darkframe

    newfile = fileframe - darktouse

    prefix = 'd'
    
    fits.writeto(prefix + filename, newfile, fileheader, overwrite=True) # finish this code too to save the FITS. Make sure it has the correct header!
    return

def norm_combine_flats(filelist):
    '''
    The function takes 1 input, a list of fits files to be combined, and normalizes
    each fits file in the list before taking the the median pixel values for each 
    pixel, and outputting a median fits file
    
    Inputs:
    filelist (array-like) - a list of strings that are filepaths to flats to be normalized and combined
    '''
    # Number of items in the filelist
    n = len(filelist)
    
    # Grab the fits data of the first file in filelist
    first_frame_data = fits.getdata(filelist[0])
    
    # Determine the shape of the files in filelist
    imsize_y, imsize_x = first_frame_data.shape
    
    # Creates a 3D array of zeros that is the size of a file in filelist X the number of files
    fits_stack = np.zeros((imsize_y, imsize_x , n))
    
    # Populates the 3D array with normalized flats from the files in filelist
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        norm_im =  im/(np.median(im))# finish new line here to normalize flats
        fits_stack[:,:,ii] = norm_im
        
    # Collapse the 3D array of normalized flat fields into a median flat frame     
    med_frame = np.median(fits_stack, axis=2)
    return med_frame

def createflats(flatdirectory, path_to_master_bias, path_to_master_dark):
    '''
    Creates master flats in all three filters
    
    Inputs:
    flatdirectory (str) - filepath to flats folder
    path_to_master_bias (str) - filepath to master bias
    path_to_master_dark (str) - filepath to master dark
    '''
    #create three directories representing the three filters, and add them to an array
    directories = []
    directories.append(flatdirectory+'/Blue/')
    directories.append(flatdirectory+'/Visual/')
    directories.append(flatdirectory+'/Red/')
    directories = np.array(directories)
    #loop through filters
    for i in range(len(directories)):
        pathx = directories[i]
        os.chdir(pathx)
        flats = glob.glob('*.fit')
        print(flats)
        #bias and dark subtract flats
        for z in flats:
            bias_subtract(z, path_to_master_bias)
        biassubtractedflats = glob.glob('b_*.fit')
        for x in biassubtractedflats:
            dark_subtract(x, path_to_master_dark)
        darkbiassubtractedflats = glob.glob('db_*.fit')
        header = fits.getheader(darkbiassubtractedflats[0])
        #normalize and combine flats
        normalizedsubtractedflats = norm_combine_flats(darkbiassubtractedflats)
        print('median of normalized and subtracted flat is' + str(np.median(normalizedsubtractedflats)))
        #save all of the master flats
        if (header['FILTER'] == 'Blue'):
            fits.writeto('Master_Flat_Bband.fit', normalizedsubtractedflats, header, overwrite=True)
        elif (header['FILTER'] == 'Visual'):
            fits.writeto('Master_Flat_Vband.fit', normalizedsubtractedflats, header, overwrite=True)
        else:
            fits.writeto('Master_Flat_Rband.fit', normalizedsubtractedflats, header, overwrite=True)
    return
            
def flatfield_correction(filepath, pathtomastervisualflatfield, pathtomasterblueflatfield, pathtomasterredflatfield):
    '''
    Applies a flatfield correction to a images within a folder
    
    Inputs:
    filepath (str) - filepath to folder
    pathtomastervisualflatfield (str) - filepath to master V flat
    pathtomasterblueflatfield (str) - filepath to master B flat
    pathtomasterredflatfield (str) - filepath to master R flat
    '''
    os.chdir(filepath)
    reducedfits = glob.glob('db_*.fit')
    #loop through images 
    for i in reducedfits:
        #check image filter
        header = fits.getheader(i)
        if header['FILTER'] == 'Visual':
            path_master_flatfield = pathtomastervisualflatfield
        elif header['FILTER'] == 'Blue':
            path_master_flatfield = pathtomasterblueflatfield
        else:
            path_master_flatfield = pathtomasterredflatfield
        
        #do the flatfield correction
        flatdata = fits.getdata(path_master_flatfield)
        data = fits.getdata(i)
        output = data/flatdata
        yeet = np.where(np.isnan(output))
        output[yeet] = 0
        #save the flat fielded image
        prefix = 'f'
        fits.writeto(prefix + i, output, header, overwrite=True)
    ff_corrected = glob.glob('fdb_*.fit')
    #sort the fully reduced images by filter
    for fitsfile in ff_corrected:
        filesorter(fitsfile, 'ReducedV', 'Visual', 'FILTER')
        filesorter(fitsfile, 'ReducedB', 'Blue', 'FILTER')
        filesorter(fitsfile, 'ReducedR', 'Red', 'FILTER')        
    

def process_images(path_to_science, master_bias_path, master_dark_path, pathtomastervisualflatfield, pathtomasterblueflatfield, pathtomasterredflatfield):
    '''
    Performs bias, dark, and flat field corrections on all science images.
    
    Inputs:
    path_to_science (str) - filepath to the lights folder
    master_bias_path (str) - filepath to master bias
    master_dark_path (str) - filepath to master dark
    pathtomastervisualflatfield (str) - filepath to master V flat
    pathtomasterblueflatfield (str) - filepath to master B flat
    pathtomasterredflatfield (str) - filepath to master R flat
    '''
    os.chdir(path_to_science)
    fitsz = glob.glob('*.fit')
    for i in fitsz:
        bias_subtract(i, master_bias_path)
    biassubtracted = glob.glob('b_*.fit')
    for i in biassubtracted:
        dark_subtract(i, master_dark_path)
    flatfield_correction(path_to_science, pathtomastervisualflatfield,pathtomasterblueflatfield, pathtomasterredflatfield)
    
    
def sortscience(path_to_science, objects=['M_52', 'G3-33', 'GD246'], directories=['ReducedB', 'ReducedV', 'ReducedR']):
    '''
    Sorts all of the science images by object within the filter folders.
    
    Inputs:
    path_to_science (str) - filepath to the lights folder
    ####### OPTIONAL #######
    objects (list of str) - list of the objects that were observed. .fit files must include this string within their filename
                            for the sorting to work
    directories (list of str) - the names of the filter folders within the science folder
    '''
    
    os.chdir(path_to_science)
    for d in directories:
        os.chdir(path_to_science+'/'+d+'/')
        for o in objects:
            objectfits = glob.glob('*'+o+'*.fit')
            for fitsfile in objectfits:
                filesorter(fitsfile, o, 'Light Frame', 'IMAGETYP')
    return

    
def cross_image(im1, im2, boxsize, ycen, xcen, **kwargs):
    """
    This function takes two images and uses a fast fourier transform to compute the correlation image.
    Determines the peak value of the correlation image and uses that to calculate the shift between images.

    Inputs:
    im1 - array-like
    im2 - array-like
    boxsize - the side length of a square that will cross correlated over
    ycen - the y coordinate center of the square
    xcen - the x coordinate center of the square

    Outputs:
    xshift, yshift - pixel shift in the x and y-directions
    """
    
    # The type cast into 'float' is to avoid overflows:
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')
    
    im1_gray[np.isinf(im1_gray)|np.isnan(im1_gray)] = np.nanmedian(im1_gray)
    im2_gray[np.isinf(im2_gray)|np.isnan(im2_gray)] = np.nanmedian(im2_gray)

    im1_gray = im1_gray[ycen-boxsize:ycen+boxsize,xcen-boxsize:xcen+boxsize]
    im2_gray = im2_gray[ycen-boxsize:ycen+boxsize,xcen-boxsize:xcen+boxsize]

    # Subtract the averages of im1_gray and im2_gray from their respective arrays -- cross-correlation
    # works better that way.
    
    # Complete the following two lines:
    im1_gray -= np.nanmean(im1_gray)
    im2_gray -= np.nanmean(im2_gray)

    # Calculate the correlation image using fast Fourier transform (FFT)
    # Note the flipping of one of the images (the [::-1]) - this is how the convolution is done.
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')
    # To determine the location of the peak value in the cross-correlated image, complete the line below,
    # using np.argmax on the correlation image:
    peak_corr_index = np.argmax(corr_image)
    # pdb.set_trace()
    # Find the peak signal position in the cross-correlation -- this gives the shift between the images.
    corr_tuple = np.unravel_index(peak_corr_index, corr_image.shape)
    
    # Calculate shifts (not cast to integer, but could be).
    xshift = corr_tuple[0] - corr_image.shape[0]/2.
    yshift = corr_tuple[1] - corr_image.shape[1]/2.

    return xshift,yshift
    
def get_ccshifts(imlist, refimage, boxsize, ycen, xcen):
    '''
    Use cross correlation to get the relative shifts between a list of images and a reference image
    
    Inputs:
    imlist (list of str) - list of filepaths of images 
    refimage (array-like) - the fits data of the reference image
    boxsize (float) - the side length of a square that will cross correlated over
    ycen (float) - the y coordinate center of the square
    xcen (float) - the x coordinate center of the square
    
    Outputs:
    shifts (array-like) - an array of tuples representing the shifts
    '''
    #os.chdir(path_to_data)
    shifts = []
    for i,im in enumerate(imlist):
        dat_compare = fits.getdata(im)
        xshift, yshift = cross_image(refimage, dat_compare, boxsize, ycen, xcen)
        shifts.append((xshift, yshift))
    return (shifts)


def stackscience(imlist, ref_image, boxsize, ycen, xcen, light_path, filter_name, object_name):
    '''
    Align and stack all of the items in a list to a certain reference image using cross-correlation
    
    Inputs:
    imlist (list of str) - list of filepaths of images 
    refimage (array-like) - the fits data of the reference image
    boxsize (float) - the side length of a square that will cross correlated over
    ycen (float) - the y coordinate center of the square
    xcen (float) - the x coordinate center of the square
    light_path (str) - filepath to the lights folder
    filter_name (str) - the filter that the images in imlist were taken with (for file naming purposes)
    object_name (str) - the name of the object in the images (for file naming purposes)
    '''
    # Make a new directory in your datadir for the new stacked fits files
    if os.path.isdir(light_path + '/Stacked') == False:
        os.mkdir(light_path + '/Stacked')
        print('\n Making new subdirectory for stacked images:', light_path + '/Stacked \n')
    
    #obtain the shifts
    shifts = get_ccshifts(imlist, ref_image, boxsize, ycen, xcen)
    #create a filepath for the stacked image
    combined_name = light_path+ '/Stacked/' + filter_name + object_name + 'stack.fit'
    #shift and combine
    shift(imlist, ref_image, shifts, 100, combined_name)
        
    
def centroidstack(path_to_science, object_name, xrange, yrange, reducedfolders=['ReducedB', 'ReducedV', 'ReducedR']):
    '''
    Aligns all the images in a directory to the first image in that directory using the centroiding method.
    
    Inputs:
    path_to_science (str) - path to the directory that contains the images to be aligned
    object_name (str) - the name of the object in the images (for file naming purposes)
    xrange (tuple) - range of x pixels to be centroided over
    yrange (tuple) - range of y pixels to be centroided over
    ###### OPTIONAL #######
    reducedfolders (list of str) - the names of the filter folders within the science folder
    '''
    for f in reducedfolders:
        os.chdir(path_to_science+'/'+f+'/'+target+'/')
        images = glob.glob('fdb_*.fit')
        b = np.nanmedian(fits.getdata(images[0]))
        shifts = align(images[0], images, xrange, yrange)
        combined_name = object_name+'_'+f+'_combined.fit'
        shift(images, shifts, 100, combined_name)
    


def centroid(fit):
  '''
  This function will calculate the centroid of an image given the fits file and
  background level. First the background level is subracted from every pixel 
  in the image. Then the centroid is calculated. 

  Input:
  fit (array-like) - 

  Output: 
  (xcen, ycen) (tuple) - represent the x pixel and y pixel of the centroid
  '''
  
  img = (fit)
  background = np.nanmedian(img)
  img = img - background

  imsize_y, imsize_x = img.shape

  rowtotal = 0
  total = 0
  for x in np.arange(imsize_x-1):
    subtotal = 0
    for y in np.arange(imsize_y-1):
      if img[x,y]>0:
        subtotal += img[x,y]
    total += subtotal
    rowtotal += x*subtotal
  xcen = rowtotal / total

  coltotal = 0
  total = 0
  for y in np.arange(imsize_y-1):
    subtotal = 0
    for x in np.arange(imsize_x-1):
      if img[x,y]>0:
        subtotal += img[x,y]
    total += subtotal
    coltotal += y*subtotal
  ycen = coltotal / total

  return (xcen, ycen)


def align(master, files, xrange, yrange):
  '''
  This function will take a master frame and a list of fits files 
  and calculate the offset between every file in the list and the master
  frame using the centroidng method.

  Inputs:
  master - filepath
  files - list of filepaths
  xrange - (x1,x2) of the region that will be centroided
  yrange - (y1,y2) of the region that will be centroided
  background - a background level in the images
    
  Outputs:
  shifts (array-like) - an array of tuples representing the shifts
  '''
  #Calculate the centroid of the master image
  master_img = fits.getdata(master)
  master_slice = master_img[yrange[0]:yrange[1], xrange[0]:xrange[1]]

  master_centroid = centroid(master_slice)

  #Make a list of fits files from the image filepaths
  imgs = []
  for f in files:
    imgs.append(fits.getdata(f))

  #Make a list of slices from the fits files
  slices = []
  for i in imgs:
    slices.append(i[yrange[0]:yrange[1], xrange[0]:xrange[1]])
  
  #Make a list of tuple shifts between the centroid of each image and the master
  shifts = []
  for s in slices:
    s_centroid = centroid(s)
    shifts.append(((s_centroid[1]-master_centroid[1]), -1*(s_centroid[0]-master_centroid[0])))

  return shifts



def shift(imlist,refimage, shifts, pad, file_path):
  '''
  This function will shift a list of images by the specified xshifts and yshifts.
  The first image in imlist should be the master image, and the first shift in shifts should be (0,0).
  It pads each image, shifts it using interpolation, and then eventually median combines the images.
    
  Inputs:
  imlist (list of str) - list of filepaths of images 
  refimage (array-like) - the fits data of the reference image
  shifts (array-like) - an array of tuples representing the shifts
  pad (float) - the number of pixels to pad each side of the image with
  file_path (str) - the name of the median combined image
  '''
  
  xshift = np.zeros(len(shifts))
  yshift = np.zeros(len(shifts))
  for i,s in enumerate(shifts):
    xshift[i] = s[0]
    yshift[i] = s[1]

  master = (refimage)
  row,col = master.shape
  shiftedims = np.zeros((len(shifts), row+(2*pad), col+(2*pad)))

  #pad and roll for each image
  for i, im in enumerate(imlist):
    name = os.path.basename(im)
    filepath2 = os.path.dirname(im)
    image = fits.getdata(im)
    row,col = image.shape

    #padded by pad pixels on each side
    padded_image = np.pad(image,pad,'constant', constant_values=-500)
    
    #replace naNs before interpolation
    padded_image = np.nan_to_num(padded_image, nan=-500, posinf=-500, neginf=-500)
    
    #shift each image by the specified values
    shifted_image = interp.shift(padded_image,(xshift[i],yshift[i]), cval=-500)

    #replace all of the cvals with nans
    shifted_image[shifted_image <= -200] = np.nan

    #write registered image
    fits.writeto(filepath2 + '/s' + name, shifted_image, header=fits.getheader(im), overwrite=True)

    #add
    shiftedims[i,:,:] = shifted_image
    
  fits.writeto(file_path, np.nanmedian(shiftedims, axis=0), header = fits.getheader(imlist[0]), overwrite=True)
  return

########################
###### PHOTOMETRY ######
########################

def bg_error_estimate(fitsfile):
    """
    This function takes a .fits file as an input, gets the data from the file, masks data greater than 3
    standard deviations from the median, replaces the masked data with NaNs, and calculates the median of this 
    error. Then, it writes two files, a background error file and a total error file, the latter considering the
    gain of the observing device (and thus Poisson noise). It outputs the raw data of the error_image.
    """
    fitsdata = fits.getdata(fitsfile) #gets the data from a .fits file
    hdr = fits.getheader(fitsfile) #saves the header as hdr
    
    # What is happening in the next step? Read the docstring for sigma_clip.
    # Answer: takes the median of the data and iterates over the data, rejecting values that are more than
    #3 standard deviations (sigma=3) away from the median. Returns an array with the same shape, with the 
    #rejected data masked 

    filtered_data = sigma_clip(fitsdata, sigma=3.,copy=False)
    
    # Summarize the following steps:
    # Takes the masked array and fills all the maxed points with a specified value, in this case a NaN
    # the background error = the square root of the above result
    # takes every nan point in the above line's output and sets it to the median of the bakground error
    bkg_values_nan = filtered_data.filled(fill_value=np.nan)
    bkg_error = np.sqrt(bkg_values_nan)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)
    
    print("Writing the background-only error image: ", fitsfile.split('.')[0]+"_bgerror.fit")
    fits.writeto(fitsfile.split('.')[0]+"_bgerror.fit", bkg_error, hdr, overwrite=True)
    
    effective_gain = 1.4 # electrons per ADU
    
    error_image = calc_total_error(fitsdata, bkg_error, effective_gain)  
    
    print("Writing the total error image: ", fitsfile.split('.')[0]+"_error.fit")
    fits.writeto(fitsfile.split('.')[0]+"_error.fit", error_image, hdr, overwrite=True)
    
    return error_image


# Star extraction function -- this function can be to also return the x and y positions to the notebook to use later:

# target_filter_xpos, target_filter_ypos = starExtractor("image.fit", nsigma_value=#, fwhm_value=#)

def starExtractor(fitsfile, nsigma_value, fwhm_value):
    """
    This function takes 3 inputs, a .fits file, a value for nsigma, and a value for FWHM. It calculates the background sigma using the median absolute standard deviation
    and uses the DAOStarFinder function to locate stars in the image. It then calculates and prints the number of stars. It produces and returns arrays of x and y centroids 
    of stars and writes these positions to a .reg file.
    """
    
    # First, check if the region file exists yet, so it doesn't get overwritten
    regionfile = fitsfile.split(".")[0] + ".reg"
     
    if os.path.exists(regionfile) == True:
        os.remove(regionfile)
        print(regionfile, "already exists in this directory. Rename or remove the .reg file and run again.")
    
    # *** Read in the data from the fits file ***
    image = fits.getdata(fitsfile)
    if 'M_52' in fitsfile:
      image = image[0:4190,0:4190]
    
    # *** Measure the median absolute standard deviation of the image: ***
    bkg_sigma = bkg_sigma = mad_std(image, ignore_nan=True)

    # *** Define the parameters for DAOStarFinder ***
    daofind = DAOStarFinder(fwhm=fwhm_value, threshold=nsigma_value*bkg_sigma)
    
    # Apply DAOStarFinder to the image
    sources = daofind(image)
    nstars = len(sources)
    print("Number of stars found in ",fitsfile,":", nstars)
    
    # Define arrays of x-position and y-position
    xpos = np.array(sources['xcentroid'])
    ypos = np.array(sources['ycentroid'])
    
    # Write the positions to a .reg file based on the input file name
    if os.path.exists(regionfile) == False:
        f = open(regionfile, 'w') 
        for i in range(0,len(xpos)):
            f.write('circle '+str(xpos[i])+' '+str(ypos[i])+' '+str(fwhm_value)+'\n')
        f.close()
        print("Wrote ", regionfile)
        return xpos, ypos # Return the x and y positions of each star as variables
        
        
# Photometry function, which returns a table of photometry values for a list of stars

# This function can be used as
# target_phot_table = measurePhotometry(file, star_xpos, star_ypos, aperture_radius, sky_inner, sky_outer, error_array)

def measurePhotometry(fitsfile, star_xpos, star_ypos, aperture_radius, sky_inner, sky_outer, error_array):
    """
    This function takes a fitsfile, positions of stars, an aperture radius, annuli specifications, and an error image
    and outputs a photometry table for the image.
    """
    # Read in the data from the fits file:
    image = fits.getdata(fitsfile)
    
    #Creates the aperture (around the stars) and the annulus (a shell around the aperture, for bkg calcs)
    #Makes a list of apertures and annuli for each star
    #starapertures = CircularAperture((star_xpos, star_ypos),r = aperture_radius)
    pos = [(star_xpos[i],star_ypos[i]) for i in np.arange(len(star_xpos))]
    starapertures = CircularAperture(pos,r = aperture_radius)
    skyannuli = CircularAnnulus(pos, r_in = sky_inner, r_out = sky_outer)
    phot_apers = [starapertures, skyannuli]
    
    # What is new about the way we're calling aperture_photometry?
    # Last time, we didn't have an error array to pass to the function. This time, our calculations will be more accurate because they consider 
    # median background AND noise 
    phot_table = aperture_photometry(image, phot_apers, error=error_array)
        
    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean = phot_table['aperture_sum_1'] / skyannuli.area
    bkg_starap_sum = bkg_mean * starapertures.area
    final_sum = phot_table['aperture_sum_0']-bkg_starap_sum
    phot_table['bg_subtracted_star_counts'] = final_sum
    
    # Calculating the mean error in the background. First, the error in the sum of values within the aperture
    #is divided by the area of the annulus to compute the mean error from the background. Sum error is computed by 
    #multiplying this by the area of the apertures - this is the total flux error within the aperture
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    bkg_sum_err = bkg_mean_err * starapertures.area
    
    # Propagating the error to find the total error: taking the square root of the sum of the squares of the Poisson noise and the background error calculated above
    phot_table['bg_sub_star_cts_err'] = np.sqrt((phot_table['aperture_sum_err_0']**2)+(bkg_sum_err**2)) 
    
    return phot_table