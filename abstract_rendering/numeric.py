import core
import numpy as np
import math

try:
    from numba import autojit
except ImportError:
    print "Error loading numba."
    autojit = lambda f: f


# ----------- Aggregators -----------
class Count(core.Aggregator):
    """Count the number of items that fall into a particular grid element."""
    out_type = np.int32
    identity = 0

    def allocate(self, width, height, glyphset, infos):
        return np.zeros((height, width), dtype=self.out_type)

    def combine(self, existing, glyph, shapecode, val):
        update = core.glyphAggregates(glyph, shapecode, 1, self.identity)
        existing[glyph[1]:glyph[3], glyph[0]:glyph[2]] += update

    def rollup(self, *vals):
        return reduce(lambda x, y: x+y,  vals)


class Sum(core.Aggregator):
    """Count the number of items that fall into a particular grid element."""
    out_type = np.int32
    identity = 0

    def allocate(self, width, height, glyphset, infos):
        return np.zeros((height, width), dtype=self.out_type)

    def combine(self, existing, glyph, shapecode, val):
        update = core.glyphAggregates(glyph, shapecode, val, self.identity)
        existing[glyph[1]:glyph[3], glyph[0]:glyph[2]] += update

    def rollup(self, *vals):
        return reduce(lambda x, y: x+y,  vals)


# -------------- Shaders -----------------
class Floor(core.Shader):
    def shade(self, grid):
        return np.floor(grid)


class Interpolate(core.Shader):
    """Interpolate between two numbers.
         Projects the input values between the low and high values passed.
         The Default is 0 to 1.
         Empty values are preserved (default is np.nan).
    """
    def __init__(self, low=0, high=1, empty=np.nan):
        self.low = low
        self.high = high
        self.empty = empty

    def shade(self, grid):
        # TODO: Gracefully handle if the whole grid is empty
        mask = (grid == self.empty)
        min = grid[~mask].min()
        max = grid[~mask].max()
        span = float(max-min)
        percents = (grid-min)/span
        return self.low + (percents * (self.high-self.low))


class Power(core.Shader):
    """Raise to a power.    Power may be fracional."""
    def __init__(self, pow):
        self.pow = pow

    def shade(self, grid):
        return np.power(grid, self.pow)


class Cuberoot(Power):
    def __init__(self):
        super(Cuberoot, self).__init__(1/3.0)


class Sqrt(core.Shader):
    def shade(self, grid):
        return np.sqrt(grid)


class Spread(core.PixelShader):
    """Spreads the values out in a regular pattern.

         TODO: Currently only does square spread.  Extend to other shapes.
         TODO: Restricted to numbers right now...implement corresponding thing
               for categories...might be 'generic'
    """

    def __init__(self, size):
        self.size = size

    def makegrid(self, grid):
        return np.zeros_like(grid)

    def pixelfunc(self, grid, x, y):
        minx = max(0, x-math.floor(self.size/2.0))
        maxx = x+math.ceil(self.size/2.0)
        miny = max(0, y-math.floor(self.size/2.0))
        maxy = y+math.ceil(self.size/2.0)

        parts = grid[minx:maxx, miny:maxy]
        return parts.sum()


class BinarySegment(core.Shader):
    """
    Paint all pixels with aggregate value above divider one color
    and below the divider another.

    TODO: Extend so out can be something other than colors
    """
    in_type = (1, np.number)
    out_type = (4, np.int32)

    def __init__(self, low, high, divider):
        self.high = high
        self.low = low
        self.divider = float(divider)

    def shade(self, grid):
        (width, height) = grid.shape[0], grid.shape[1]
        outgrid = np.ndarray((width, height, 4), dtype=np.uint8)
        mask = (grid >= self.divider)
        outgrid[mask] = self.high
        outgrid[~mask] = self.low
        return outgrid


class InterpolateColors(core.Shader):
    """
    High-definition interpolation between two colors.
    Zero-values are treated separately from other values.

    TODO: Remove log, just provide a shader to pre-transform the values

    * low -- Color ot use for lowest value
    * high -- Color to use for highest values
    * log -- Set to desired log base to use log-based interpolation
                     (use True or "e" for base-e; default is False)
    * reserve -- color to use for empty cells
    """
    in_type = (1, np.number)
    out_type = (4, np.int32)

    def __init__(self,
                 low, high,
                 log=False,
                 reserve=core.Color(255, 255, 255, 255),
                 empty=np.nan):
        self.low = low
        self.high = high
        self.reserve = reserve
        self.log = log
        self.empty = empty

    # TODO: there are issues with zeros here....
    def _log(self,  grid):
        mask = (grid == self.empty)
        min = grid[~mask].min()
        max = grid[~mask].max()

        grid[mask] = 1
        if (self.log == 10):
            min = math.log10(min)
            max = math.log10(max)
            span = float(max-min)
            percents = (np.log10(grid)-min)/span
        elif (self.log == math.e or self.log):
            min = math.log(min)
            max = math.log(max)
            span = float(max-min)
            percents = (np.log(grid)-min)/span
        elif (self.log == 2):
            min = math.log(min, self.log)
            max = math.log(max, self.log)
            span = float(max-min)
            percents = (np.log2(grid)-min)/span
        else:
            rebase = math.log(self.log)
            min = math.log(min, self.log)
            max = math.log(max, self.log)
            span = float(max-min)
            percents = ((np.log(grid)/rebase)-min)/span

        grid[mask] = 0

        colorspan = (self.high.asarray().astype(np.int32)
                     - self.low.asarray().astype(np.int32))

        outgrid = (percents[:, :, np.newaxis]
                   * colorspan[np.newaxis, np.newaxis, :]
                   + self.low.asarray()).astype(np.uint8)
        outgrid[mask] = self.reserve
        return outgrid

    def _linear(self, grid):
        mask = (grid == self.empty)
        min = grid[~mask].min()
        max = grid[~mask].max()
        span = float(max-min)
        percents = (grid-min)/span

        colorspan = (self.high.asarray().astype(np.int32)
                     - self.low.asarray().astype(np.int32))
        outgrid = (percents[:, :, np.newaxis]
                   * colorspan[np.newaxis, np.newaxis, :]
                   + self.low.asarray()).astype(np.uint8)
        outgrid[mask] = self.reserve
        return outgrid

    def shade(self, grid):
        if (self.log):
            return self._log(grid)
        else:
            return self._linear(grid)
