import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.path
import matplotlib.pyplot as plt
import openslide
import cv2
import skimage.morphology
import PIL.Image as Image
import random

msk_aperio_20x_mpp = 0.50185

########TISSUE

def normalize_msk20x(slide):
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    mult = msk_aperio_20x_mpp/mpp
    level = 0
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
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_thumbsize(slide,size):
    w = int(np.round(slide.dimensions[0]/size))
    h = int(np.round(slide.dimensions[1]/size))
    return w, h

def threshold(slide,size):
    w = int(np.round(slide.dimensions[0]/size))
    h = int(np.round(slide.dimensions[1]/size))
    thumbnail = slide.get_thumbnail((w,h))
    thumbnail = thumbnail.resize((w,h))
    cimg, bwimg = image2array(thumbnail)
    #calc std on color image
    std = np.std(cimg,axis=-1)
    ## remove black dots ##
    _,tmp = cv2.threshold(bwimg,20,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    tmp = cv2.dilate(tmp,kernel,iterations = 1)
    bwimg[tmp==255] = 255
    bwimg = cv2.GaussianBlur(bwimg,(5,5),0)
    t,bwimg = cv2.threshold(bwimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bwimg = 255-bwimg
    bwimg[std<5] = 0
    return bwimg, t

def tissue_grid(slide,patch_size=224):
    _, mult = normalize_msk20x(slide)
    img,th = threshold(slide,patch_size*mult)
    size_x = img.shape[1]
    size_y = img.shape[0]
    offset_x = np.floor((slide.dimensions[0]/float(patch_size*mult)-size_x)*float(patch_size*mult))
    offset_y = np.floor((slide.dimensions[1]/float(patch_size*mult)-size_y)*float(patch_size*mult))
    add_x = np.linspace(0,offset_x,size_x).astype(int)
    add_y = np.linspace(0,offset_y,size_y).astype(int)
    #list of sample pixels
    w = np.where(img>0)
    grid = list(zip((w[1]*float(patch_size*mult)+add_x[w[1]]).astype(int),(w[0]*float(patch_size*mult)+add_y[w[0]]).astype(int)))
    return grid

def plot_tissue_grid(slide,patch_size=224):
    import matplotlib.patches as patches
    _, mult = normalize_msk20x(slide)
    grid = tissue_grid(slide,patch_size)
    thumb = slide.get_thumbnail((np.round(slide.dimensions[0]/50.),np.round(slide.dimensions[1]/50.)))
    ps = []
    for tup in grid:
        ps.append(patches.Rectangle(
            (tup[0]/50., tup[1]/50.), patch_size*mult/50., patch_size*mult/50., fill=False,
            edgecolor="red"
        ))
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(thumb)
    for p in ps:
        ax.add_patch(p)
    plt.show()

########XML ANNOTATIONS

def get_annotations(path):
    tree = ET.parse(path)
    root = tree.getroot()
    #get annotation nodes
    annotations = root[0].findall('Annotation')
    #get label for the annotations
    labels = [int(ann.attrib['PartOfGroup'].strip('_')) for ann in annotations]
    #get coordinates of annotations
    xy = []
    for ann in annotations:
        coords = np.zeros((len(ann[0]),2))
        for i,c in enumerate(ann[0]):
            #each vertex
            coords[i,0] = float(c.get('X'))
            coords[i,1] = float(c.get('Y'))
        xy.append(coords)
    return xy, labels

def get_bbox(coords):
    xmax, ymax = np.max(np.vstack(tuple(coords)), axis=0).astype(int)
    xmin, ymin = np.min(np.vstack(tuple(coords)), axis=0).astype(int)
    return xmax, ymax, xmin, ymin

def get_grid(coords, step=224):
    xmax, ymax, xmin, ymin = get_bbox(coords)
    xgrid = np.arange(xmin,xmax,step)
    ygrid = np.arange(ymin,ymax,step)
    xgrid, ygrid = np.meshgrid(xgrid,ygrid)
    grid = np.hstack((xgrid.flatten().reshape(-1,1),ygrid.flatten().reshape(-1,1)))
    return grid

def istumor(coords, labels, grid):
    good = []
    bad = []
    for i in range(len(coords)):
        path = matplotlib.path.Path(coords[i])
        inside = path.contains_points(grid, radius=1e-9)
        if labels[i]==2:
            bad.append(inside)
        else:
            good.append(inside)
    good = np.vstack(tuple(good)).any(0)
    if len(bad)>0:
        bad = np.vstack(tuple(bad)).any(0)
        good = np.logical_xor(good, bad)
    return good

def isnegative(coords, grid):
    good = []
    for i in range(len(coords)):
        path = matplotlib.path.Path(coords[i])
        inside = path.contains_points(grid, radius=1e-9)
        good.append(inside)
    good = ~np.vstack(tuple(good)).any(0)
    return good

#def extract_annotations(path, patchsize=224, step=112):
#    coords, labels = get_annotations(path)
#    grid = get_grid(coords, step)
#    inside = istumor(coords, labels, grid)
#    outcoords = grid[inside]
#    #offset to top left corner
#    outcoords = outcoords - (patchsize/2.)
#    return zip(outcoords[:,0],outcoords[:,1])

def extract_annotations(path, slide, patchsize=224, step=112):
    coords, labels = get_annotations(path)
    grid = get_grid(coords, step)
    inside = istumor(coords, labels, grid)
    outpos = grid[inside]
    #offset to top left corner
    _, mult = normalize_msk20x(slide)
    outpos = outpos - int(patchsize*mult/2.)
    outpos = outpos.astype(int)
    #sampling the negative tissue
    tissue = tissue_grid(slide, patch_size=patchsize)
    tissue = np.array(tissue)
    outside = isnegative(coords, tissue)
    outneg = tissue[outside]
    return zip(outpos[:,0],outpos[:,1]), zip(outneg[:,0],outneg[:,1])

def plot_annotations(path, pathsize=224, step=112, figsize=20):
    coords, labels = get_annotations(path)
    grid = get_grid(coords, step)
    inside = istumor(coords, labels, grid)
    #plot
    xmax, ymax, xmin, ymin = get_bbox(coords)
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    for coord in coords:
        patch = plt.Polygon(coord, zorder=0, fill=False, lw=2, color='g')
        ax.add_patch(patch)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
    plt.scatter(grid[:,0],grid[:,1], c=inside.astype(float), cmap="RdBu_r", vmin=-.1, vmax=1.2)
    plt.show()

def plot_annotations_grid(path,slide,patchsize=224,step=112,maxnum=None):
    import matplotlib.patches as patches
    _, mult = normalize_msk20x(slide)
    gridpos, gridneg = extract_annotations(path, slide, patchsize=patchsize, step=step)
    if maxnum is not None:
        gridpos = random.sample(gridpos, maxnum)
        gridneg = random.sample(gridneg, maxnum)
    thumb = slide.get_thumbnail((np.round(slide.dimensions[0]/50.),np.round(slide.dimensions[1]/50.)))
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(thumb)
    if len(gridpos) > 0:
        ps = []
        for tup in gridpos:
            ps.append(patches.Rectangle(
                (tup[0]/50., tup[1]/50.), patchsize*mult/50., patchsize*mult/50., fill=False,
                edgecolor="red"
            ))
        for p in ps:
            ax.add_patch(p)
    if len(gridneg) > 0:
        ps = []
        for tup in gridneg:
            ps.append(patches.Rectangle(
                (tup[0]/50., tup[1]/50.), patchsize*mult/50., patchsize*mult/50., fill=False,
                edgecolor="blue"
            ))
        for p in ps:
            ax.add_patch(p)
    plt.show()

###TEST
#path = '/lila/data/fuchs/projects/challenges/CAMELYON16/training/lesion_annotations/tumor_003.xml'
#slidename = '/lila/data/fuchs/projects/challenges/CAMELYON16/training/tumor/tumor_003.tif'
#slide = openslide.OpenSlide(slidename)
#plot_tissue_grid(slide)
#grid = tissue_grid(slide)

#plot_annotations_grid(path,slide)
#gridpos, gridneg = extract_annotations(path, slide, patchsize=224, step=112)
#plot_annotations(path)
