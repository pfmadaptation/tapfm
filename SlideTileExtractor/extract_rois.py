import json
import requests
import os
import matplotlib.path
import matplotlib.pyplot as plt
import openslide
import numpy as np
import SlideTileExtractor.roi_morphology_utils as morphology

MAX_PIXEL_DIFFERENCE = 0.1

def get_rois_from_url(url):
    response = requests.get(url)
    if response.status_code == 200 and response.text != '[]':
        rois = json.loads(response.text)
    else:
        rois = None
    return rois

def parse_rois(rois):
    labels = []
    coords = []
    bboxs = []
    for roi in rois:
        labels.append(int(roi['class']))
        xs = [int(float(x)) for x in roi['xs']]
        ys = [int(float(x)) for x in roi['ys']]
        bbox = get_bbox(xs, ys)
        coords.append(list(zip(xs, ys)))
        bboxs.append(bbox)
    return coords, labels, bboxs

def get_bbox(xs, ys):
    xmax, xmin = max(xs), min(xs)
    ymax, ymin = max(ys), min(ys)
    return xmax, ymax, xmin, ymin

def bbox_to_size(bbox, pad=0):
    xmax, ymax, xmin, ymin = bbox
    origin = (xmin - pad, ymin - pad)
    w = xmax - xmin + 2*pad
    h = ymax - ymin + 2*pad
    return origin, w, h

def bbox_from_coords(coords):
    xs , ys = list(zip(*coords))
    xmax, ymax, xmin, ymin = max(xs), max(ys), min(xs), min(ys)
    return xmax, ymax, xmin, ymin

def join_bboxs(bboxs):
    xmax, ymax, xmin, ymin = None, None, None, None
    for xmax_, ymax_, xmin_, ymin_ in bboxs:
        xmax = max(xmax, xmax_)
        ymax = max(ymax, ymax_)
        xmin = min(i for i in [xmin, xmin_] if i is not None)
        ymin = min(i for i in [ymin, ymin_] if i is not None)
    return xmax, ymax, xmin, ymin

def get_grid(bbox, step=224):
    xmax, ymax, xmin, ymin = bbox
    xgrid = np.arange(xmin,xmax,step)
    ygrid = np.arange(ymin,ymax,step)
    xgrid, ygrid = np.meshgrid(xgrid,ygrid)
    grid = np.hstack((xgrid.flatten().reshape(-1,1),ygrid.flatten().reshape(-1,1)))
    return grid

def isinside(coord, grid):
    path = matplotlib.path.Path(coord)
    is_inside = path.contains_points(grid, radius=1e-9)
    return grid[is_inside,:]

def find_level(slide, res, patchsize=224):
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    downsample = res/mpp
    for i in range(slide.level_count)[::-1]:
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

def find_size(slide, mpp, tilesize):
    maxres = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    mult = mpp / maxres
    size = np.round(mult * tilesize) / tilesize
    if abs(size - tilesize) < MAX_PIXEL_DIFFERENCE * tilesize:
        size = tilesize
    return size

def generate_url(user, project_id, dirs, slide_id):
    if dirs:
        dirs = ';'.join(dirs)+';'
    else:
        dirs = ''
    url = "http://slides-res.mskcc.org/slides/{}@mskcc.org/{};{}{}/getSVGLabels/roi".format(user,project_id,dirs,slide_id)
    return url

def extract_annotations(user, project_id, dirs, slide_id, target_label, step=20, tile_size=224, offset=True, delta=0):
    url = generate_url(user, project_id, dirs, slide_id)
    rois = get_rois_from_url(url)
    if rois is None:
        return None
    else:
        output = []
        coords, labels, bboxs = parse_rois(rois)
        for coord, label, bbox in zip(coords, labels, bboxs):
            # Erode/dilate
            if delta > 0:
                # Dilate
                coord = morphology.dilateroi(coord, delta)
            elif delta < 0:
                # Erode
                coord = morphology.eroderoi(coord, -delta)
            if label == -1:
                # Extact from tissue outside the annotations
                pass
            if label == target_label:
                grid = get_grid(bbox, step)
                grid = isinside(coord, grid)
                # Offset to top left corner
                if offset:
                    grid = grid - int(tile_size/2.)
                grid = grid.astype(int)
                output.extend(list(zip(grid[:,0], grid[:,1])))
        return output

def plot_annotations(prefix, user, project_id, dirs, slide_id, target_label, step=20, tile_size=224, delta=0, fig_size=20, level=2):
    slide = openslide.OpenSlide(os.path.join(prefix, slide_id))
    centers = extract_annotations(user, project_id, dirs, slide_id, target_label, step, tile_size, offset=False, delta=delta)
    if centers is None:
        raise Exception('No ROIs found')
    bbox = bbox_from_coords(centers)
    origin, w, h = bbox_to_size(bbox, pad=0)
    
    data = np.array(centers)
    downsample = slide.level_downsamples[level]
    window = slide.read_region(origin, level, (int(w/downsample),int(h/downsample)))
    
    fig, ax = plt.subplots(figsize=(fig_size,fig_size))
    plt.imshow(window)
    plt.scatter((data[:,0]-origin[0])/downsample,(data[:,1]-origin[1])/downsample, marker='.')
    plt.show()


#TEST
#from SlideTileExtractor import extract_rois
#prefix = '/lila/data/fuchs/projects/lung/impacted/'
#slide_col = 1
#dirs_col = 0
#project_id = 91
#user = 'vanderbc'
#dirs = ['p-0009508-t01-im5']
#slide_id = '398995.svs'
#target_label = 0
#step = 50
#tile_size = 224
#delta = 100

#extract_rois.extract_annotations(user, project_id, dirs, slide_id, target_label, step=20, tile_size=224, offset=True, delta=delta)
#extract_rois.plot_annotations(prefix, user, project_id, dirs, slide_id, target_label, step=20, tile_size=224, delta=delta, fig_size=20, level=2)
