# SlideTileExtractor
This module allows for the extraction of tiles from slides. It leverages openslide, so it is hopefully slide format agnostic.
The are a series of submodules that perform different tasks:
* `extract_tissue.py`: it is the main submodule. It can extract tiles from tissue and it has an option to extract tiles from the intersection of tissue and a BMP annotation. Note: it will extract from any annotation regardless of class.
* `extract_BMPannotations.py`: submodule dedicated to extract tiles from within BMP annotations.
* `extract_rois.py`: submodule dedicated to exract tiles from polygon (a.k.a. rois) annotations.
* `extract_ASAPannotations.py`: module dedicated to extract tiles from XML annotations from the ASAP software. Useful for the CAMELYON dataset.

All submodules handle tile extraction at different resolutions in microns per pixel (MPP). With MSK Aperio scanners, 20x magnification is equivalent to 0.5 MPP.

## Tissue tiles (`extract tissue`)
The main function to use is `make_sample_grid`. It accepts the following parameters:
* `slide`: `openslide` object;
* `patch_size`: tile size;
* `mpp`: requested resolution in MPP;
* `power`: resolution in objective power. Will be converted to MPP automatically;
* `min_cc_size`: minimum size of connected components considered tissue;
* `max_ratio_size`: deprecated parameter, has no effect;
* `dilate`: after thresholding it will dilate the tissue mask;
* `erode`: after thresholding it will erode the tissue mask;
* `prune`: checks whether each found tile contains enough tissue. *Note: this is very slow*;
* `overlap`: how much overlap between consecutive tiles in a non overlapping grid;
* `maxn`: if not `None`, is the maximum number of sampled tiles *per class*;
* `bmp`: if not `None`, is the path to the BMP annotation file from which extract the tissue tiles;
* `oversample`: to extract the tiles in a grid from the highest resolution regardless of the requested resolution. Good for slides that have little tissue.
It returns:
* list of coordinate tuples (x,y).

Another important function is `find_level`, which given a slide object, the requested resolution and tile size, calculate:
* from which level you need to extract tiles;
* what scale you need to apply to your extracted tiles to get the requested resolution.

*Note: if the appropriate tile size is less than 10% different than the working tile size (e.g. for a 224x224px tile, 22px tolerance around 224), no scaling will be applied to avoid interpolation artifacts.*

The function `plot_extraction` allows to show the result of the tile extraction.

Example usage:
```python
from SlideTileExtractor.extract_tissue import extract tissue, find_level, plot_extraction
```

## BMP annotations (`extract_BMPannotations`)
The main function to use is `extract_bmp_grid`. It accepts the following parameters:
* `path` to bmp file;
* `stride` with which extract tiles in a grid;
* `size` of the tiles (important for resolution); 
* `maxn`, if not `None` is the maximum number of sampled tiles *per class*.
It returns:
* list of coordinate tuples (x,y);
* list of respective classes as defined in the viewer project.

The function `find_size` with parameters slide object, requested mpp and requested tile size, returns the size to use to achieve tiles for a specific resolution. *Note: this module can handle arbitrary resolutions. Just use the proper tile size needed to achieve the desired resolution.*

The function `find_level` with similar parameters as above, returns:
* from which level you need to extract tiles;
* what scale you need to apply to your extracted tiles to get the requested resolution.
*Note: if the appropriate tile size is less than 10% different than the working tile size (e.g. for a 224x224px tile, 22px tolerance around 224), no scaling will be applied to avoid interpolation artifacts.*

There are plotting functions to test whether the module is working as expected:
* `plot`: it plots the tiles over the BMP file;
* `plot_bmp_grid`: it plots the tiles over the actual slide.

Example usage:
```python
from SlideTileExtractor.extract_BMPannotations import plot, plot_bmp_grid, extract_bmp_grid
path = '422437.svs_labels_lem@mskcc.org.bmp'
slidename = './2016-12-13/422437.svs'
slide = openslide.OpenSlide(slidename)
stride = 20
size = 224
maxn = 100
plot(path, stride, size, maxn)
plot_bmp_grid(path, slide, stride, size, maxn=maxn, downsample=20.)
grid, targets = extract_bmp_grid(path, stride, size, maxnum=None)
```

## Polygon/ROI annotations (`extract_rois`)
The main functions is `extract_annotations`. It accepts the following parameters:
* `user`: the msk email name of the annotator;
* `project_id`: the project ID (as in the viewer);
* `dirs`: list of directories if present;
* `slide_id`: the slide name as shown in the viewer;
* `target_label`: the class for which labels are extracted. *Note: only rois labeld with this class will be extracted*;
* `step`: the stride;
* `tile_size`: the tile size (important for resoltion). 
It returns:
* list of coordinate tuples (x,y) if there are annotations for that slide/user or `None` otherwise.

The function `find_size` with parameters slide object, requested mpp and requested tile size, returns the size to use to achieve tiles for a specific resolution. *Note: this module can handle arbitrary resolutions. Just use the proper tile size needed to achieve the desired resolution.*

The function `find_level` with similar parameters as above, returns:
* from which level you need to extract tiles;
* what scale you need to apply to your extracted tiles to get the requested resolution.
*Note: if the appropriate tile size is less than 10% different than the working tile size (e.g. for a 224x224px tile, 22px tolerance around 224), no scaling will be applied to avoid interpolation artifacts.*

The function `plot_annotations` plots the extraction.
