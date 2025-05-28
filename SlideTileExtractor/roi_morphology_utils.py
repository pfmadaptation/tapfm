import numpy as np
import matplotlib.path
import matplotlib.pyplot as plt
from collections import OrderedDict 
from matplotlib.backends.backend_pdf import PdfPages
from SlideTileExtractor import extract_tissue
import pdb
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from pointpats.centrography import hull

def get_point(window, delta, orig, mode):
    a, b, c = window
    ba = a - b
    bc = c - b
    angle_ba = np.arccos(ba[0]/np.linalg.norm(ba)) * np.sign(ba[1]+1e-16)
    angle_bc = np.arccos(bc[0]/np.linalg.norm(bc)) * np.sign(bc[1]+1e-16)
    alpha = (angle_ba + angle_bc)/2
    dx = np.cos(alpha)
    dy = np.sin(alpha)
    dplus = b+np.array([dx,dy])
    dminus = b-np.array([dx,dy])
    if mode == 'inside':
        if isinside(orig, dplus.reshape(1,-1)):
            return b+(np.array([dx,dy])*delta).round().astype(int)
        else:
            return b-(np.array([dx,dy])*delta).round().astype(int)
    elif mode == 'outside':
        if isinside(orig, dplus.reshape(1,-1)):
            return b-(np.array([dx,dy])*delta).round().astype(int)
        else:
            return b+(np.array([dx,dy])*delta).round().astype(int)

def angle(a, b, c):
    ba = a - b
    bc = c - b
    theta = np.arccos(np.clip(np.dot(ba,bc) / ( np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1))
    if np.isnan(theta):
        pdb.set_trace()
    return np.degrees(theta)

def isinside(coord, grid):
    path = matplotlib.path.Path(coord)
    is_inside = path.contains_points(grid, radius=1e-9)
    return is_inside

def eroderoi(coords, delta):
    orig = list(coords)
    # Remove duplicates
    orig = list(OrderedDict.fromkeys(orig))
    # Simplify
    #orig = simplify(orig, tolerance)
    # Pad roi
    orig.append(orig[0])
    orig.append(orig[1])
    orig = np.array(orig)
    traces = []
    traces_paths = []
    points = []
    for i in range(len(orig)-3):
        window = orig[i:i+3,:]
        good = get_point(window, delta, orig, 'inside')
        points.append(good.flatten())
        #if not good.shape[0] == 0:
        #    if len(points)>1:
        #        # Makes sure that same point is not appended twice
        #        if not np.all(points[-1]==good):
        #            # If angle between last 2 points is too small then break segment
        #            theta = angle(points[-2], points[-1], good.flatten())
        #            if theta < thetamin:
        #                traces.append(np.stack(points))
        #                traces_paths.append(matplotlib.path.Path(np.stack(points)))
        #                points = []
        #            points.append(good.flatten())
        #            # If the current segment intersects previous segments then break segment
        #            current_path = matplotlib.path.Path(np.stack(points))
        #            intersects = False
        #            for p, path in enumerate(traces_paths):
        #                if path.intersects_path(current_path, filled=False):
        #                    # Break old segment
        #                    t1, t2 = break_trace(traces[p], good.flatten())
        #                    del traces[p]
        #                    del traces_paths[p]
        #                    traces.insert(p, t1)
        #                    traces.insert(p+1, t2)
        #                    traces_paths.insert(p, matplotlib.path.Path(t1))
        #                    traces_paths.insert(p+1, matplotlib.path.Path(t2))
        #                    # Break new segment
        #                    points = points[:-1]
        #                    traces.append(np.stack(points))
        #                    traces_paths.append(matplotlib.path.Path(np.stack(points)))
        #                    points = []
        #                    points.append(good.flatten())
        #                    break
        #    else:
        #        # First 2 steps
        #        points.append(good.flatten())
    #traces.append(np.stack(points))
    traces = np.stack(points)
    # Remove small traces
    #traces = [x for x in traces if len(x)>sizemin]
    # Remove traces that are inside zone
    traces = remove_bad_points(traces, np.array(coords), delta)
    # Flatten points
    #traces = [item for sublist in traces for item in sublist]
    if len(traces) < 3:
        return None
    else:
        #return np.stack(traces)
        return traces

def dilateroi(coords, delta):
    orig = list(coords)
    # Remove duplicates
    orig = list(OrderedDict.fromkeys(orig))
    # Simplify
    #orig = simplify(orig, tolerance)
    # Pad roi
    orig.append(orig[0])
    orig.append(orig[1])
    orig = np.array(orig)
    traces = []
    traces_paths = []
    points = []
    for i in range(len(orig)-3):
        window = orig[i:i+3,:]
        good = get_point(window, delta, orig, 'outside')
        points.append(good.flatten())
        #if not good.shape[0] == 0:
        #    if len(points)>1:
        #        # Makes sure that same point is not appended twice
        #        if not np.all(points[-1]==good):
        #            # If angle between last 2 points is too small then break segment
        #            theta = angle(points[-2], points[-1], good.flatten())
        #            if theta < thetamin:
        #                traces.append(np.stack(points))
        #                traces_paths.append(matplotlib.path.Path(np.stack(points)))
        #                points = []
        #            points.append(good.flatten())
        #            # If the current segment intersects previous segments then break segment
        #            current_path = matplotlib.path.Path(np.stack(points))
        #            for p, path in enumerate(traces_paths):
        #                if path.intersects_path(current_path, filled=False):
        #                    # Break old segment
        #                    t1, t2 = break_trace(traces[p], good.flatten())
        #                    del traces[p]
        #                    del traces_paths[p]
        #                    traces.insert(p, t1)
        #                    traces.insert(p+1, t2)
        #                    traces_paths.insert(p, matplotlib.path.Path(t1))
        #                    traces_paths.insert(p+1, matplotlib.path.Path(t2))
        #                    # Break new segment
        #                    points = points[:-1]
        #                    traces.append(np.stack(points))
        #                    traces_paths.append(matplotlib.path.Path(np.stack(points)))
        #                    points = []
        #                    points.append(good.flatten())
        #                    break
        #    else:
        #        # First 2 steps
        #        points.append(good.flatten())
    #traces.append(np.stack(points))
    traces = np.stack(points)
    # Remove small traces
    #traces = [x for x in traces if len(x)>sizemin]
    # Remove traces that are inside zone
    #traces = remove_bad_traces(traces, np.array(coords), delta)
    traces = remove_bad_points(traces, np.array(coords), delta)
    # Flatten points
    #traces = [item for sublist in traces for item in sublist]
    if len(traces) < 3:
        return None
    else:
        #return np.stack(traces)
        return traces

def bbox(arr):
    return arr[:,0].min(), arr[:,0].max(), arr[:,1].min(), arr[:,1].max()

def break_trace(trace, point):
    # trace is 2d np array
    # point is 1d np.array
    dists = np.linalg.norm(trace - point.reshape(1,-1), axis=1)
    index = np.argsort(dists)[:2].max()
    t1 = trace[:index]
    t2 = trace[index:]
    return t1, t2

def traces_to_edgepoints(traces):
    ts = []
    ls = []
    for t in traces:
        ts.append(np.stack((t[0],t[-1])))
        ls.append(np.linalg.norm(t[0]-t[-1]))
    return ts, ls

def remove_bad_traces(traces, orig, delta):
    # traces is a list of 2d arrays
    # orig is a 2d array
    ts = []
    for t in traces:
        dists = t.reshape(t.shape[0],1,-1)-orig.reshape(1,orig.shape[0],orig.shape[1])
        dists = np.linalg.norm(dists, axis=2)
        n = np.any(dists < delta*0.99, axis=1).sum()
        if n < 2:
            ts.append(t)
    return ts

def remove_bad_points(trace, orig, delta):
    # trace is a 2d array
    # orig is a 2d array
    dists = trace.reshape(trace.shape[0],1,-1)-orig.reshape(1,orig.shape[0],orig.shape[1])
    dists = np.linalg.norm(dists, axis=2)
    idxs = np.where(~np.any(dists < delta*0.99, axis=1))[0]
    return trace[idxs]

def simplify(orig, tolerance=0.2):
    newpath = []
    path = matplotlib.path.Path(orig)
    path.simplify_threshold = tolerance
    for vertex, code in path.iter_segments(simplify=True):
        newpath.append(vertex)
    newpath = newpath[:-1]
    return newpath

def plot_erosion(coords, delta, step, title=None, pdf=None):
    # Get eroded rois
    traces = eroderoi(coords, delta, 0.1, sizemin=3)
    # Make grid of points
    xmin, xmax, ymin, ymax = bbox(traces)
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymin, ymax, step))
    grid = np.array(zip(xx.flatten(), yy.flatten()))
    # Make path of eroded points
    path = matplotlib.path.Path(traces)
    # Inside points
    inside = path.contains_points((grid), radius=1e-9)
    # plot
    plt.plot(np.array(coords)[:,0], np.array(coords)[:,1])
    plt.plot(traces[:,0], traces[:,1])
    plt.scatter(grid[inside][:,0], grid[inside][:,1])
    plt.axes().set_aspect('equal', 'datalim')
    if title is not None:
        plt.title(title)
    if pdf is not None:
        pdf.savefig()
    else:
        plt.show()

def plot_dilation(coords, delta, step, title=None, pdf=None):
    # Get eroded rois
    traces = dilateroi(coords, delta, 0.1, sizemin=3)
    # Get grid
    grid = get_grid(traces, step)
    # plot
    plt.plot(np.array(coords)[:,0], np.array(coords)[:,1])
    plt.plot(traces[:,0], traces[:,1])
    plt.scatter(grid[:,0], grid[:,1])
    plt.axes().set_aspect('equal', 'datalim')
    if title is not None:
        plt.title(title)
    if pdf is not None:
        pdf.savefig()
    else:
        plt.show()

def PolyArea(trace):
    x, y = trace[:,0], trace[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

#def get_tissue_traces(slide, dilate=0, erode=0, mult=4):
#    # Get tissue tiles
#    tissue = np.array(extract_tissue.make_sample_grid(slide, patch_size=224, mpp=0.5, mult=mult)).astype(np.uint32)
#    # HC to divide in subregions (e.g. needle biopsies)
#    Z = linkage(tissue, method='single', metric='cityblock')
#    labels = fcluster(Z, 500, criterion='distance')
#    nregs = np.unique(labels).size
#    for i in range(nregs):
#        h = hull(tissue[labels==i+1, :])

def get_grid(trace, step):
    # Make grid of points
    xmin, xmax, ymin, ymax = bbox(trace)
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymin, ymax, step))
    grid = np.array(list(zip(xx.flatten(), yy.flatten())))
    # Make path of eroded points
    path = matplotlib.path.Path(trace)
    # Inside points
    inside = path.contains_points(grid, radius=1e-9)
    return grid[inside]

def get_grid_donut(trace_out, trace_in, step):
    # Make grid of points
    xmin, xmax, ymin, ymax = bbox(trace_out)
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymin, ymax, step))
    grid = np.array(list(zip(xx.flatten(), yy.flatten())))
    # Make path of eroded points
    path_out = matplotlib.path.Path(trace_out)
    path_in = matplotlib.path.Path(trace_in)
    # Points within traces
    cond_out = path_out.contains_points(grid, radius=1e-9)
    cond_in = ~path_in.contains_points(grid, radius=1e-9)
    return grid[cond_out & cond_in]

def get_tissue_inside(trace, slide, step=56, mult=4):
    # Get the grid from the trace
    area = PolyArea(trace)
    if area/step**2 > 5e4:
        step = int(np.ceil(np.sqrt(area/5e4)))
    grid = get_grid(trace, step).astype(np.uint32)
    # Get tissue tiles
    tissue = np.array(extract_tissue.make_sample_grid(slide, patch_size=224, mpp=0.5, mult=mult)).astype(np.uint32)
    # Only consider tissue tiles around the BBOX of the grid
    xmin, xmax, ymin, ymax = bbox(grid)
    tissue = tissue[(tissue[:,0]>=xmin-224) & (tissue[:,0]<=xmax+224) & (tissue[:,1]>=ymin-224) & (tissue[:,1]<=ymax+224),:]
    if len(tissue)>0 and len(grid)>0:
        # Offset to center of tile
        tissue += 224//2
        # Get distance of each center in the grid with each tissue point
        dists = grid.reshape(grid.shape[0],1,-1) - tissue.reshape(1,tissue.shape[0],tissue.shape[1])
        dists = np.linalg.norm(dists, axis=2)
        dists = dists.min(axis=1)
        idxs = np.where(~(dists > int(224/2*(2**0.5))+1))[0]
        return grid[idxs]
    else:
        return None

def get_tissue_donut(trace_out, trace_in, slide, step=56, mult=4):
    # Get the grid from the trace
    area = PolyArea(trace_out) - PolyArea(trace_in)
    if area/step**2 > 5e4:
        step = int(np.ceil(np.sqrt(area/5e4)))
    grid = get_grid_donut(trace_out, trace_in, step).astype(np.uint32)
    # Get tissue tiles
    tissue = np.array(extract_tissue.make_sample_grid(slide, patch_size=224, mpp=0.5, mult=mult)).astype(np.uint32)
    # Only consider tissue tiles around the BBOX of the grid
    xmin, xmax, ymin, ymax = bbox(grid)
    tissue = tissue[(tissue[:,0]>=xmin-224) & (tissue[:,0]<=xmax+224) & (tissue[:,1]>=ymin-224) & (tissue[:,1]<=ymax+224),:]
    if len(tissue)>0 and len(grid)>0:
        # Offset to center of tile
        tissue += 224//2
        # Get distance of each center in the grid with each tissue point
        dists = grid.reshape(grid.shape[0],1,-1) - tissue.reshape(1,tissue.shape[0],tissue.shape[1])
        dists = np.linalg.norm(dists, axis=2)
        dists = dists.min(axis=1)
        idxs = np.where(~(dists > int(224/2*(2**0.5))+1))[0]
        return grid[idxs]
    else:
        return None

#def check(trace, slide):
#    '''
#    Given roi and slide, check 
#    '''
#    tissue = np.array(extract_tissue.make_sample_grid(slide, patch_size=224, mpp=0.5, mult=4, centerpixel=True))
#    # Get distance of roi point with each tissue point


### Example usage
#rois = extract_rois.get_rois_from_url('https://slides-res.mskcc.org/slides/vanderbc@mskcc.org/91;p-0000208-t02-im5;1142939.svs/getSVGLabels/roi')
#coords, labels, bboxs = extract_rois.parse_rois(rois)
#delta = 300
#traces = eroderoi(coords[0], delta, 0.1, sizemin=3)
#traces = dilateroi(coords[0], delta, 0.1, sizemin=3)
#plot_erosion(coords[0], delta, 50)
#plot_dilation(coords[0], delta, 50)

#import extract_rois
#rois = extract_rois.get_rois_from_url('https://slides-res.mskcc.org/slides/vanderbc@mskcc.org/91;p-0010454-t01-im5;408452.svs/getSVGLabels/roi')
#coords, labels, bboxs = extract_rois.parse_rois(rois)
#delta=300
#trace = dilateroi(coords[1], delta, 0.1, sizemin=3)
#plot_dilation(coords[1], delta, 50)

#Example with non tissue regions
#import extract_rois
#import extract_tissue
#import roi_morphology_utils as morphology
#import openslide
#rois = extract_rois.get_rois_from_url('https://slides-res.mskcc.org/slides/vanderbc@mskcc.org/91;p-0009928-t01-im5;404115.svs/getSVGLabels/roi')
#coords, labels, bboxs = extract_rois.parse_rois(rois)
#delta=300
#trace = morphology.dilateroi(coords[1], delta, 0.1, sizemin=3)
#slide = openslide.OpenSlide('/lila/data/fuchs/projects/lung/impacted/404115.svs')
#tissue = morphology.get_tissue(trace, slide, step=56)

#rois = extract_rois.get_rois_from_url('https://slides-res.mskcc.org/slides/vanderbc@mskcc.org/91;p-0006139-t02-im6;755801.svs/getSVGLabels/roi')
#tg = extract_tissue.make_sample_grid(slide, mult=1)
#extract_tissue.plot_extraction(slide)
