from __future__ import print_function, division
import blaze as blz
import numpy as np
import abstract_rendering.glyphset as glyphset
import abstract_rendering.core as ar


class Glyphset(glyphset.Glyphset):
    def __init__(self, table, xcol, ycol, valcol, vt=(0, 0, 1, 1)):
        self._table = table
        self.xcol = xcol
        self.ycol = ycol
        self.valcol = valcol
        self.vt = vt
        self.shaper = glyphset.ToPoint(glyphset.idx(xcol),
                                       glyphset.idx(ycol))

        self.table = blz.transform(table,
                                   xcol=(table[xcol] * vt[2]) + vt[0],
                                   ycol=(table[ycol] * vt[3]) + vt[1])

    def points(self):
        return self.table[[self.xcol, self.ycol]]

    def data(self):
        return self.table[self.valcol]

    def project(self, vt):
        nvt = (self.vt[0]+vt[0],
               self.vt[1]+vt[1],
               self.vt[2]*vt[2],
               self.vt[3]*vt[3])
        return Glyphset(self._table, self.xcol, self.ycol, self.valcol, vt=nvt)

    def bounds(self):
        xmax = blz.compute(self.table[self.xcol].max())
        xmin = blz.compute(self.table[self.xcol].min())
        ymax = blz.compute(self.table[self.ycol].max())
        ymin = blz.compute(self.table[self.ycol].min())
        return (xmin, ymin, xmax-xmin, ymax-ymin)


class Count(ar.Aggregator):
    "Blaze sepcific implementation of the count aggregator"

    def aggregate(self, glyphset, info, screen):
        points = glyphset.table
        xcol = glyphset.xcol
        ycol = glyphset.ycol

        finite = blz.transform(points,
                               x=blz.floor(points[xcol]),
                               y=blz.floor(points[ycol]),
                               info=points[xcol].map(lambda a: 1, schema='{info: int32}'))

        sparse = blz.by(finite[[xcol, ycol]],
                        finite['info'].count())

        return to_numpy(sparse, screen, '__info_count')

    def rollup(self, *vals):
        return reduce(lambda x, y: x+y,  vals)


class Sum(ar.Aggregator):
    "Blaze sepcific implementation of the sum aggregator"

    def __init__(self, info):
        self.infotype = info
        super(ar.Aggregator, self).__init__()

    def aggregate(self, glyphset, info, screen):
        points = glyphset.table
        xcol = glyphset.xcol
        ycol = glyphset.ycol

        schema = "{info: %s}" % self.infotype
        finite = blz.transform(points,
                               x=blz.floor(points[xcol]),
                               y=blz.floor(points[ycol]),
                               info=points[glyphset.valcol].map(info, schema=schema))

        sparse = blz.by(finite[[xcol, ycol]],
                        finite['info'].sum())
        return to_numpy(sparse, screen, 'info_sum')

    def rollup(self, *vals):
        return reduce(lambda x, y: x+y,  vals)


class CountCategories(ar.Aggregator):
    def __init__(self, info):
        self.infotype = info
        super(ar.Aggregator, self).__init__()

    def aggregate(self, glyphset, info, screen):
        points = glyphset.table
        xcol = glyphset.xcol
        ycol = glyphset.ycol

        schema = "{info: %s}" % self.infotype
        infos = points[glyphset.valcol].map(info, schema=schema)
        cats = infos.distinct()

        finite = blz.transform(points,
                               x=blz.floor(points[xcol]),
                               y=blz.floor(points[ycol]),
                               info=infos)

        sparse = blz.by(finite[[xcol, xcol, 'info']],
                        finite['info'].count())

        items = []
        for cat in blz.compute(cats):
            subset = sparse[sparse['info'] == cat]
            items.append(to_numpy(subset[['x', 'y', 'info_count']],
                                  screen,
                                  'info_count'))

        rslt = np.dstack(items)
        return rslt

    def rollup(self, *vals):
        """NOTE: Assumes co-registration of categories..."""
        return reduce(lambda x, y: x+y,  vals)


def to_numpy(sparse, screen=None, values='__info'):
    """
    Convert a blaze table to a numpy arary.
    Assumes table schema format is [x,y,val]

    TODO: Add screen_origin so a subset of the space can sliced out easily
    """
    if not screen:
        screen = (blz.compute(sparse['x'].max()) + 1,
                  blz.compute(sparse['y'].max()) + 1)
    else:
        # Just things that fit on the screen
        sparse = sparse[sparse['x'] < screen[0] and sparse['y'] < screen[1]]

    (width, height) = screen

    xx = blz.into(np.ndarray, sparse.__x).astype(np.int32)
    yy = blz.into(np.ndarray, sparse.__y).astype(np.int32)
    vals = blz.into(np.ndarray, sparse[values])

    dense = np.zeros((height, width), dtype=vals.dtype)
    dense[yy, xx] = vals
    return dense


def load_csv(file, xc, yc, vc, **kwargs):
    """
    Produce point-based glyphs in a blaze-backed glyphset from a csv file.

    * File - name of a csv file to load
    * schema - scheam of the source file
    * xc - name of the column to use for x values
    * yc - name of column to use for y values
    * vc - name of column to use for info values
    """

    csv = blz.CSV(file, **kwargs)
    t = blz.Table(csv)
    return Glyphset(t, xc, yc, vc)


# Perhaps a blaze-grid?  Would require grid to be a wrapper around np array...
#      would provide also place for category mappings to live...
# Perhaps a blaze-transfer.  Could work with cellFunc already (probably)....
#    maybe add that to all transfers, and the numpy version is "special"....
#    is this like a numpy ufunc...maybe all of the shaders should be ufuncs....
# If a blaze-transfer comes about, then the to_numpy in the aggregators
#    is no longer required and a transfer that does it should be provided
