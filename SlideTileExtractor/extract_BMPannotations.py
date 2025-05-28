import numpy as np
import matplotlib.path
import matplotlib.pyplot as plt
import openslide
import cv2
import skimage.morphology
from skimage import measure
import PIL.Image as Image
import random
from scipy import ndimage
Image.MAX_IMAGE_PIXELS = 100000000000#to avoid warning

MAX_PIXEL_DIFFERENCE = 0.1

def loadbmp(name):
    im = Image.open(name,'r')
    bbox = im.getbbox()
    im = im.crop(bbox)
    (wbbox,hbbox) = im.size
    bbmap = np.frombuffer(im.tobytes(), dtype='uint8').reshape(hbbox,wbbox)
    #bbmap=np.array(im.getdata(),dtype='uint8').reshape(hbbox,wbbox)
    print('Bounding box width: {}'.format(wbbox))
    print('Bounding box height: {}'.format(hbbox))
    print('Loaded {} pixels'.format(wbbox*hbbox))
    return bbmap, (bbox[0],bbox[1])

def getrois(bmparray, offset):
    labelmap, nlabels = measure.label(bmparray,background=0,return_num=True)
    labels = np.arange(nlabels)+1
    values = [np.unique(bmparray[labelmap==x])[0] for x in labels]
    rois = []
    starts = []
    for i in range(nlabels):
        slice_r, slice_c = ndimage.find_objects(labelmap==labels[i])[0]
        roi = np.copy(labelmap[slice_r,slice_c])
        roi[roi!=labels[i]] = 0
        rois.append(roi)
        starts.append((slice_c.start+offset[0], slice_r.start+offset[1]))
    print('Found {} connected components'.format(nlabels))
    return rois, starts, values

def getgridfromroi(roi, offset, stride, size):
    rh, rw = roi.shape
    x = np.arange(0, rw, stride)
    y = np.arange(0, rh, stride)
    gridx, gridy = np.meshgrid(x, y)
    check = roi[gridy.flatten(), gridx.flatten()] > 0
    grid = list(zip((gridx.flatten()[check]+offset[0]-size/2).astype(int), (gridy.flatten()[check]+offset[1]-size/2).astype(int)))
    return grid

def makegrid(rois, offsets, targets, stride, size, maxn):
    grid = []
    t = []
    for roi,off,target in zip(rois, offsets, targets):
        g = getgridfromroi(roi, off, stride, size)
        grid.extend(g)
        t.extend([target]*len(g))
    if maxn:
        grid, t = samplegrid(grid, t, maxn)
    print('Found {} tiles'.format(len(grid)))
    return grid, t

def samplegrid(grid, targets, maxn):
    us = np.unique(targets)
    outg = []
    outt = []
    for u in us:
        outg.extend(random.sample(list(np.array(grid)[np.array(targets)==u]), min(maxn,(np.array(targets)==u).sum())))
        outt.extend([u]*min(maxn,len(grid)))
    return outg, outt
        
def extract_bmp_grid(path, stride, size, maxn=None):
    bmp, bboffset = loadbmp(path)
    rois, starts, targets = getrois(bmp, bboffset)
    grid, targets = makegrid(rois, starts, targets, stride, size, maxn)
    return grid, targets

def plot(path, stride, size, maxn=None):
    import matplotlib.patches as patches
    cmap = plt.get_cmap('Spectral')
    bmp, bboffset = loadbmp(path)
    rois, starts, targets = getrois(bmp, (0,0))
    grid, targets = makegrid(rois, starts, targets, stride, size, maxn)
    colord = getcolordict(targets)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(bmp)
    if len(grid) > 0:
        ps = []
        for tup,t in zip(grid,targets):
            ps.append(patches.Rectangle((tup[0], tup[1]), size, size, fill=False, edgecolor=cmap(int(colord[t]))))
        for p in ps:
            ax.add_patch(p)
    plt.show()

def plot_bmp_grid(path, slide, stride, size, maxn=None, downsample=20.):
    import matplotlib.patches as patches
    cmap = plt.get_cmap('Spectral')
    grid, targets = extract_bmp_grid(path, stride, size, maxn)
    colord = getcolordict(targets)
    thumb = slide.get_thumbnail((np.round(slide.dimensions[0]/downsample),np.round(slide.dimensions[1]/downsample)))
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(thumb)
    if len(grid) > 0:
        ps = []
        for tup,t in zip(grid,targets):
            ps.append(patches.Rectangle(
                (tup[0]/downsample, tup[1]/downsample), size/downsample, size/downsample, fill=False,
                edgecolor=cmap(int(colord[t]))
            ))
        for p in ps:
            ax.add_patch(p)
    plt.show()

def getcolordict(targets):
    ts = list(np.unique(targets))
    d = {}
    for i,t in enumerate(ts):
        d[t] = float(i)/(len(ts)-1)*255
    return d

def find_size(slide, mpp, tilesize):
    maxres = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    mult = mpp / maxres
    size = np.round(mult * tilesize) / tilesize
    if abs(size - tilesize) < MAX_PIXEL_DIFFERENCE * tilesize:
        size = tilesize
    return size

def find_level(slide,res,patchsize=224):
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    downsample = res/mpp#maxres/res
    for i in range(slide.level_count)[::-1]:
        #if abs(slide.level_downsamples[i]-downsample)<0.009 or downsample>slide.level_downsamples[i]:
        if abs(downsample / slide.level_downsamples[i] * patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize or downsample > slide.level_downsamples[i]:
            level = i
            mult = downsample / slide.level_downsamples[level]
            break
    else:
        raise Exception('Requested resolution ({} mpp) is too high'.format(res))
    #move mult to closest pixel
    mult = np.round(mult*patchsize)/patchsize
    if abs(mult*patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize:
        mult = 1.
    return level, mult

###TEST
#path = '/lila/data/fuchs/projects/BCC/2016-12-13/maps/422437.svs_labels_lem@mskcc.org.bmp'
#slidename = '/lila/data/fuchs/projects/BCC/2016-12-13/422437.svs'
#slide = openslide.OpenSlide(slidename)
#stride = 20
#size = 224
#maxn = 100
#plot(path, stride, size, maxn)
#grid, targets = extract_bmp_grid(path, stride, size)
#plot_bmp_grid(path, slide, stride, size, maxn=maxn, downsample=20.)
