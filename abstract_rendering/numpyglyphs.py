from __future__ import print_function, division
import numpy as np
from scipy.ndimage.filters import convolve
from fast_project import _projectRects
import abstract_rendering.glyphset as glyphset
import abstract_rendering.core as ar
import abstract_rendering.util as util


class Glyphset(glyphset.Glyphset):
    # TODO: Default data is list of None (?)
    def __init__(self, points, data, vt=(0, 0, 1, 1)):
        self._points = points
        self._data = data
        self.vt = vt
        self.shaper = glyphset.ToPoint(glyphset.idx(0), glyphset.idx(1))

        if not is_identity_transform(vt):
            self.projected = np.empty_like(points, dtype=np.int32)
            _projectRects(vt, points, self.projected)
            #points must be at least 1 wide for general compatability
            self.projected[:, 2] = self.projected[:, 0]+1
            self.projected[:, 3] = self.projected[:, 1]+1
        else:
            self.projected = points

    def data(self):
        return self._data

    def points(self):
        return self.projected

    def project(self, vt):
        """
        Project the points found in the glyphset with to the transform.

        vt -- convert canvas space to pixel space [tx,ty,sx,sy]
        returns a new glyphset with projected points and associated info values
        """
        nvt = (self.vt[0]+vt[0],
               self.vt[1]+vt[1],
               self.vt[2]*vt[2],
               self.vt[3]*vt[3])
        return Glyphset(self._points, self._data, nvt)

    def bounds(self):
        (xmax, ymax, _, _) = self.projected.max(axis=0)
        (xmin, ymin, _, _) = self.projected.min(axis=0)
        bounds = (xmin, ymin, xmax-xmin, ymax-ymin)
        return bounds


class PointCount(ar.Aggregator):
    def aggregate(self, glyphset, info, screen):
        sparse = glyphset.points()
        dense = np.histogram2d(sparse[:, 0], sparse[:, 1], screen)
        return dense[0].T

    def rollup(self, *vals):
        return reduce(lambda x, y: x+y,  vals)


class PointCountCategories(ar.Aggregator):
    def aggregate(self, glyphset, info, screen):
        points = glyphset.points()
        cats = glyphset.data().max() 

        (width, height) = screen
        dims = (height, width, cats+1) 
        
        (xmin, ymin, _, _) = points.min(axis=0)
        (xmax, ymax, _, _) = points.max(axis=0)
        ranges = ((ymin, ymax), (xmin, xmax), (0, cats+1)) #Need cats+1 to get the breaks correct

        data = np.hstack([np.fliplr(points[:, 0:2]), glyphset.data()[:, np.newaxis]])
        dense = np.histogramdd(data, bins=dims, range=ranges)
        return dense[0]

    def rollup(self, *vals):
        """NOTE: Assumes co-registration of categories..."""
        return reduce(lambda x, y: x+y,  vals)


class Spread(ar.CellShader):
    """
    Spreads the values out in a regular pattern.
    Spreads categories inside their category plane (not between planes).
    
    * factor : How far in each direction to spread
    """
    def __init__(self, factor=1):
        self.factor = factor

    def shade(self, grid):
        kShape = (self.factor*2+1, self.factor*2+1) 
        k = np.ones(kShape)

        if len(grid.shape) == 3:
            out = np.empty_like(grid)
            cats = grid.shape[2]
            for cat in xrange(cats):
                out[:, :, cat] = convolve(grid[:, :, cat], k, mode='constant', cval=0.0)
        else:
            out = convolve(grid, k, mode='constant', cval=0.0)

        return out


class Log10(ar.CellShader):
    def shade(self, grid):
        mask = (grid == 0)
        out = np.log10(grid)
        out[mask] = 0
        return out


def is_identity_transform(vt):
    return vt == (0, 0, 1, 1)

def code_categories(values, cats, defcat=-1):
    """Encode the items in values as numeric indicies.
    The index will equal the order in "cats".

    * values : List of values to encode
    * cats : List of categoires. 
             If none, is generated from values
    * defcat: Default category **index**.  
              Defaults to len(cats)
    """
    if (isinstance(values, util.EmptyList)
            or values is None): 
        return util.EmptyList()
    
    data = np.asanyarray(values)
    if cats is None:
        cats = np.unique(data)   # returns sorted values!!
    codes = dict(zip(cats, xrange(len(cats))))
    defcat = defcat if defcat >=0 else len(cats)
    coded = [codes.get(cat, defcat) for cat in data]  # TODO: Remove this SEQUENTIAL op
    coded = np.array(coded)

    return coded


def load_csv(filename, skip, xc, yc, vc):
    """Turn a csv file into a glyphset.

    This is a fairly naive regulary-expression based parser
    (it doesn't handle quotes, blank lines or much else).
    It is useful for getting simple datasets into the system.
    """
    import re
    source = open(filename, 'r')
    points = []
    data = []

    for i in range(0, skip):
        source.readline()

    for line in source:
        line = re.split("\s*,\s*", line)
        x = float(line[xc].strip())
        y = float(line[yc].strip())
        v = float(line[vc].strip()) if vc >= 0 else 1
        g = [x, y, 0, 0]
        points.append(g)
        data.append(v)

    source.close()
    return Glyphset(np.array(points, order="F"), np.array(data))


def load_hdf(filename, node, xc, yc, vc=None, cats=None):
    """
    Load a node from an HDF file.

    filename : HDF file to load
    node: Path to relevant HDF table
    xc: Name/index of the x column
    yc: Name/index of the y column
    vc: Name/index of the value column (if applicable)
    cats: List of expected categories. 
        If cats is an empty list, a coding will be automatically generated
        Any value not on the list will be assigned category equal to list lenght
        Ignored if vc is not supplied.
    """
    import pandas as pd
    table = pd.read_hdf(filename, node)
    points = table[[xc, yc]]
    a = np.zeros((len(points), 4), order="F")
    a[:, :2] = points

   
    if vc:
        data = table[vc] 
        coded = code_categories(data, cats)
    else:
        coded = None

    print("Loaded %d items" % len(a))

    # Is this copy needed?  After all, projection makes a copy too...
    #    maybe the input just needs to be CLOSE to a numpy array
    return Glyphset(a, coded)
