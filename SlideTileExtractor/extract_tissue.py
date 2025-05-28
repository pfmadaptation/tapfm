from SlideTileExtractor import utils
import cv2
import numpy as np
import openslide
from skimage.morphology import binary_erosion, binary_dilation, label, dilation, square, skeletonize
from skimage.filters import threshold_otsu
import PIL.Image as Image
import random
import pdb

msk_aperio_20x_mpp = 0.5
MAX_PIXEL_DIFFERENCE = 0.1 # difference must be within 10% of image size

def normalize_msk20x(slide):
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    mult = msk_aperio_20x_mpp/mpp
    level = 0
    return level, mult

def power2mpp(power):
    return msk_aperio_20x_mpp*20./power

def find_level(slide,res,patchsize=224):
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    downsample = res/mpp#maxres/res
    for i in range(slide.level_count)[::-1]:
        if abs(downsample / slide.level_downsamples[i] * patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize or downsample > slide.level_downsamples[i]:
        #if slide.level_downsamples[i] <= (downsample+downsample*0.001):
        #if abs(slide.level_downsamples[i]-downsample)<0.009 or downsample>slide.level_downsamples[i]:
            level = i
            mult = downsample / slide.level_downsamples[level]
            break
    else:
        raise Exception('Requested reolution ({} mpp) is too high'.format(res))
    #move mult to closest pixel
    mult = np.round(mult*patchsize)/patchsize
    if abs(mult*patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize:
        mult = 1.
    return level, mult

def image2array(img):
    if img.__class__.__name__=='Image':
        if img.mode=='RGB':
            img=np.array(img)
            r,g,b = np.rollaxis(img, axis=-1)
            img=np.stack([r,g,b],axis=-1)
        elif img.mode=='RGBA':
            img=np.array(img)
            r,g,b,a = np.rollaxis(img, axis=-1)
            img=np.stack([r,g,b],axis=-1)
        else:
            sys.exit('Error: image is not RGB slide')
    img=np.uint8(img)
    return img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def is_sample(img,threshold=0.9,ratioCenter=0.1,wholeAreaCutoff=0.5,centerAreaCutoff=0.9):
    nrows,ncols=img.shape
    timg=cv2.threshold(img, 255*threshold, 1, cv2.THRESH_BINARY_INV)
    kernel=np.ones((5,5),np.uint8)
    cimg=cv2.morphologyEx(timg[1], cv2.MORPH_CLOSE, kernel)
    crow=np.rint(nrows/2).astype(int)
    ccol=np.rint(ncols/2).astype(int)
    drow=np.rint(nrows*ratioCenter/2).astype(int)
    dcol=np.rint(ncols*ratioCenter/2).astype(int)
    centerw=cimg[crow-drow:crow+drow,ccol-dcol:ccol+dcol]
    if (np.count_nonzero(cimg)<nrows*ncols*wholeAreaCutoff) & (np.count_nonzero(centerw)<4*drow*dcol*centerAreaCutoff):
        return False
    else:
        return True

def threshold(slide, size, res, maxres, mult=1):
    w = int(np.round(slide.dimensions[0]*1./(size*res/maxres))) * mult
    h = int(np.round(slide.dimensions[1]*1./(size*res/maxres))) * mult
    thumbnail = slide.get_thumbnail((w,h))
    thumbnail = thumbnail.resize((w,h))
    img_c = image2array(thumbnail)
    #calc std on color image
    std = np.std(img_c, axis=-1)
    #image to bw
    #img_g = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    img_g = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)
    
    # Detect markers
    marker = utils.detect_marker(img_c, maxres/res*mult)
    #img_g[np.nonzero(marker)] = np.mean(img_g)
    
    # Otsu
    img_g = cv2.GaussianBlur(img_g, (5, 5), 0)
    # New Otsu with skimage and masked arrays
    if marker is not None:
        masked = np.ma.masked_array(img_g, marker>0)
        t = threshold_otsu(masked.compressed())
        img_g = cv2.threshold(img_g, t, 255, cv2.THRESH_BINARY)[1]
    
    else:
        t, img_g = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Exclude marker
    if marker is not None:
        img_g = cv2.subtract(~img_g, marker)
        #img_g = cv2.erode(img_g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
        #img_g = cv2.dilate(img_g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    else:
        img_g = 255 - img_g
    
    # Remove grays
    img_g[std<5] = 0
    
    # Rescale
    if mult>1:
        img_g = img_g.reshape(h//mult, mult, w//mult, mult).max(axis=(1, 3))

    return img_g, t

def remove_black_ink(img_g, th=50, delta=50):
    '''
    image in gray scale
    returns mask where ink is positive
    th=50 and delta=50 was chosen based on some slides
    '''
    dist = np.clip(img_g - float(th), 0, None)
    mask = dist < delta
    if mask.sum() > 0:
        mask_s = skeletonize(mask)
        d = int(np.round(0.1 * mask.sum() / mask_s.sum()))
        mask = dilation(mask, square(2*d+1))
        return mask
    else:
        return None
 
def filter_regions(img,min_size):
    l, n = label(img, return_num=True)
    for i in range(1,n+1):
        #filter small regions
        if l[l==i].size < min_size:
            l[l==i] = 0
    return l

def add(overlap):
    return np.linspace(0,1,overlap+1)[1:-1]

def add2offset(img, slide, patch_size, mpp, maxmpp):
    size_x = img.shape[1]
    size_y = img.shape[0]
    offset_x = np.floor((slide.dimensions[0]*1./(patch_size*mpp/maxmpp)-size_x)*(patch_size*mpp/maxmpp))
    offset_y = np.floor((slide.dimensions[1]*1./(patch_size*mpp/maxmpp)-size_y)*(patch_size*mpp/maxmpp))
    add_x = np.linspace(0,offset_x,size_x).astype(int)
    add_y = np.linspace(0,offset_y,size_y).astype(int)
    return add_x, add_y

def addoverlap(w, grid, overlap, patch_size, mpp, maxmpp, img, offset=0):
    o = (add(overlap)*(patch_size*mpp/maxmpp)).astype(int)
    ox,oy = np.meshgrid(o,o)
    connx = np.zeros(img.shape).astype(bool)
    conny = np.zeros(img.shape).astype(bool)
    connd = np.zeros(img.shape).astype(bool)
    connu = np.zeros(img.shape).astype(bool)
    connx[:,:-1] = img[:,1:]
    conny[:-1,:] = img[1:,:]
    connd[:-1,:-1] = img[1:,1:]
    connu[1:,:-1] = img[:-1,1:] & ( ~img[1:,1:] | ~img[:-1,:-1] )
    connx = connx[w]
    conny = conny[w]
    connd = connd[w]
    connu = connu[w]
    extra = []
    for i,(x,y) in enumerate(grid):
        if connx[i]: extra.extend(zip(o+x-offset,np.repeat(y,overlap-1)-offset))
        if conny[i]: extra.extend(zip(np.repeat(x,overlap-1)-offset,o+y-offset))
        if connd[i]: extra.extend(zip(ox.flatten()+x-offset,oy.flatten()+y-offset))
        if connu[i]: extra.extend(zip(x+ox.flatten()-offset,y-oy.flatten()-offset))
    return extra

def get_tissue_trace(slide):
    maxmpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    img, th = threshold(slide, 224, 0.5, maxmpp, 4)
    img[img>0]=1
    eroded = binary_erosion(img)
    img = img - eroded
    add_x, add_y = add2offset(img, slide, 224, 0.5, maxmpp)
    w = np.where(img>0)
    grid = list(zip((w[1]*(224*0.5/maxmpp)+add_x[w[1]]).astype(int),(w[0]*(224*0.5/maxmpp)+add_y[w[0]]).astype(int)))
    offset = int(0.5 / maxmpp * 224 // 2)
    grid = [(x[0] + offset, x[1] + offset) for x in grid]
    return np.array(grid)

def make_sample_grid(slide,patch_size=224,mpp=0.5,power=None,min_cc_size=10,max_ratio_size=10,dilate=False,erode=False,prune=False,overlap=1,maxn=None,bmp=None, oversample=False, mult=1, centerpixel=False):
    '''
    Script that given an openslide object return a list of tuples
    in the form of (x,y) coordinates for patch extraction of sample patches.
    It has an erode option to make sure to get patches that are full of tissue.
    It has a prune option to check if patches are sample. It is slow.
    If bmp is given, it samples from within areas of the bmp that are nonzero.
    If oversample is True, it will downsample for full resolution regardless of what resolution is requested.
    mult is used to increase the resolution of the thumbnail to get finer  tissue extraction
    '''
    if power:
        mpp = power2mpp(power)

    maxmpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    if oversample:
        img, th = threshold(slide,patch_size,maxmpp,maxmpp,mult)
    else:
        img, th = threshold(slide,patch_size,mpp,maxmpp,mult)
    
    if bmp:
        bmplab = Image.open(bmp)
        thumbx, thumby = img.shape
        bmplab = bmplab.resize((thumby, thumbx), Image.ANTIALIAS)
        bmplab = np.array(bmplab)
        bmplab[bmplab>0] = 1
        img = np.logical_and(img, bmplab)
    
    img = filter_regions(img,min_cc_size)
    img[img>0]=1
    if erode:
        img = binary_erosion(img)
    if dilate:
        img = binary_dilation(img)

    if oversample:
        add_x, add_y = add2offset(img, slide, patch_size, maxmpp, maxmpp)
    else:
        add_x, add_y = add2offset(img, slide, patch_size, mpp, maxmpp)
    
    #list of sample pixels
    w = np.where(img>0)

    #grid=zip(w[1]*patch_size,w[0]*patch_size)
    if oversample:
        offset = int(0.5 * patch_size * ((mpp/maxmpp) - 1))
        grid = list(zip((w[1]*(patch_size)+add_x[w[1]]-offset).astype(int),(w[0]*(patch_size)+add_y[w[0]]-offset).astype(int)))
    else:
        grid = list(zip((w[1]*(patch_size*mpp/maxmpp)+add_x[w[1]]).astype(int),(w[0]*(patch_size*mpp/maxmpp)+add_y[w[0]]).astype(int)))

    #connectivity
    if overlap > 1:
        if oversample:
            extra = addoverlap(w, grid, overlap, patch_size, maxmpp, maxmpp, img, offset=offset)
            grid.extend(extra)
        else:
            extra = addoverlap(w, grid, overlap, patch_size, mpp, maxmpp, img)
            grid.extend(extra)

    # center pixel offset
    if centerpixel:
        offset = int(mpp / maxmpp * patch_size // 2)
        grid = [(x[0] + offset, x[1] + offset) for x in grid]

    #prune squares
    if prune:
        level, mult = find_level(slide,mpp,maxmpp)
        psize = int(patch_size*mult)
        truegrid = []
        for tup in grid:
            reg = slide.read_region(tup,level,(psize,psize))
            if mult != 1:
                reg = reg.resize((224,224),Image.BILINEAR)
            reg = image2array(reg)
            if is_sample(reg,th/255,0.2,0.4,0.5):
                truegrid.append(tup)
    else:
        truegrid = grid
    
    #sample if maxn
    if maxn:
        truegrid = random.sample(truegrid, min(maxn, len(truegrid)))

    return truegrid

def make_hires_map(slide, pred, grid, patch_size, mpp, maxmpp, overlap):
    '''
    Given the list of predictions and the known overlap it gives the hires probability map
    '''
    W = slide.dimensions[0]
    H = slide.dimensions[1]
    w = int(np.round(W*1./(patch_size*mpp/maxmpp)))
    h = int(np.round(H*1./(patch_size*mpp/maxmpp)))

    newimg = np.zeros((h*overlap,w*overlap))-1
    offset_x = np.floor((W*1./(patch_size*mpp/maxmpp)-w)*(patch_size*mpp/maxmpp))
    offset_y = np.floor((H*1./(patch_size*mpp/maxmpp)-h)*(patch_size*mpp/maxmpp))
    add_x = np.linspace(0,offset_x,w).astype(int)
    add_y = np.linspace(0,offset_y,h).astype(int)
    for i,(xgrid,ygrid) in enumerate(grid):
        yindx = int(ygrid/(patch_size*mpp/maxmpp))
        xindx = int(xgrid/(patch_size*mpp/maxmpp))
        y = np.round((ygrid-add_y[yindx])*overlap/(patch_size*mpp/maxmpp)).astype(int)
        x = np.round((xgrid-add_x[xindx])*overlap/(patch_size*mpp/maxmpp)).astype(int)
        newimg[y,x] = pred[i]
    return newimg

def make_hires_map_stride(slide, pred, grid, stride):
    '''
    Given the list of predictions and the stride it gives the hires probability map
    Grid ndarray specify the center pixel of a tile
    '''
    W = slide.dimensions[0]
    H = slide.dimensions[1]
    w = int(round(W*1./stride))
    h = int(round(H*1./stride))

    # Scale grid to pixels
    ngrid = np.floor(grid.astype(float) / stride).astype(int)

    # Make image
    newimg = np.zeros((h,w))-2

    # Add tissue
    tissue = threshold_stride(slide, stride)
    newimg[tissue>0] = -1

    # paint predictions
    for i in range(len(ngrid)):
        x, y = ngrid[i]
        newimg[y,x] = pred[i]

    return newimg

def threshold_stride(slide, stride):
    W = slide.dimensions[0]
    H = slide.dimensions[1]
    w = int(np.ceil(W*1./stride))
    h = int(np.ceil(H*1./stride))
    thumbnail = slide.get_thumbnail((w,h))
    thumbnail = thumbnail.resize((w,h))
    img = image2array(thumbnail)
    #calc std on color image
    std = np.std(img,axis=-1)
    #image to bw
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## remove black dots ##
    _,tmp = cv2.threshold(img,20,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    tmp = cv2.dilate(tmp,kernel,iterations = 1)
    img[tmp==255] = 255
    img = cv2.GaussianBlur(img,(5,5),0)
    t,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = 255-img
    img[std<5] = 0
    return img

def plot_extraction(slide,patch_size=224,mpp=0.5,power=None,min_cc_size=10,max_ratio_size=10,dilate=False,erode=False,prune=False,overlap=1,maxn=None,bmp=None,oversample=False,mult=1,save=''):
    '''Script that shows the result of applying the detector in case you get weird results'''
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if save:
        plt.switch_backend('agg')

    #maxres = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    maxmpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    grid = make_sample_grid(slide,patch_size,mpp=mpp,power=power,min_cc_size=min_cc_size,max_ratio_size=max_ratio_size,dilate=dilate,erode=erode,prune=prune,overlap=overlap,maxn=maxn,bmp=bmp,oversample=oversample,mult=mult)
    thumb = slide.get_thumbnail((np.round(slide.dimensions[0]/50.),np.round(slide.dimensions[1]/50.)))
    

    ps = []
    for tup in grid:
        ps.append(patches.Rectangle(
            (tup[0]/50., tup[1]/50.), patch_size/50.*(mpp/maxmpp), patch_size/50.*(mpp/maxmpp), fill=False,
            edgecolor="purple"
        ))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(thumb)
    for p in ps:
        ax.add_patch(p)
    if save:
        plt.savefig(save)
    else:
        plt.show()

#kernel=np.ones((5,5),np.uint8)
#cimg=cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)

#out=[]
#for tup in grid:
#    out.append(sum([1 if x==tup else 0 for x in grid]))
#np.unique(np.array(out))
#TESTS
#from SlideTileExtractor import extract_tissue
#import openslide
#slide = openslide.OpenSlide('/lila/data/fuchs/projects/lung/impacted/1142892.svs')
#extract_tissue.plot_extraction(slide, patch_size=224, mpp=0.5, power=None, min_cc_size=10, max_ratio_size=10, dilate=False, erode=False, prune=False, overlap=1, maxn=None, bmp=None, oversample=False, mult=1, save='')
