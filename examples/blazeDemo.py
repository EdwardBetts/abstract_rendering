#!/usr/bin/env python
"""
Draws a colormapped image plot
 - Left-drag pans the plot.
 - Mousewheel up and down zooms the plot in and out.
 - Pressing "z" brings up the Zoom Box, and you can click-drag a rectangular
   region to zoom.  If you use a sequence of zoom boxes, pressing alt-left-arrow
   and alt-right-arrow moves you forwards and backwards through the "zoom
   history".
"""
# Abstract rendering imports
import abstract_rendering.util as util
import abstract_rendering.core as core
import abstract_rendering.numeric as numeric
import abstract_rendering.categories as categories
import abstract_rendering.infos as infos
import abstract_rendering.glyphset as glyphset
import abstract_rendering.blazeglyphs as blzg
import abstract_rendering.numpyglyphs as npg

from timer import Timer

# Enthought library imports
from enable.api import Component, ComponentEditor
from traits.api import HasTraits, Instance
from traitsui.api import Item, Group, View

# Chaco imports
from chaco.api import ArrayPlotData, Plot

#===============================================================================
# # Create the Chaco plot.
#===============================================================================
def _create_plot_component():
    red = util.Color(255,0,0,255)
    green = util.Color(0,255,0,255)
    blue = util.Color(0,0,255,255)
    purple = util.Color(125,0,255,255)
    white = util.Color(255,255,255,255)
    black = util.Color(0,0,0,255)
    
    shape = glyphset.ShapeCodes.RECT
    glyphs = blzg.load_csv("../data/circlepoints.csv", "x", "y", "series", 
                            schema="{r:float32, theta:float32, x:float32, y:float32, series:int32}")

    screen = (800,600)
    vt = util.zoom_fit(screen,glyphs.bounds(), False)

    with Timer("Abstract-Render") as arTimer:   
#      image = core.render(glyphs, 
#                          infos.val(),
#                          blzg.CountCategories("int32"),
#                          npg.Spread(20) + categories.HDAlpha([red, blue, green, purple, black]),
#                          screen,
#                          vt)
      image = core.render(glyphs, 
                          infos.valAt(4,0),
                          blzg.Count(), 
                          numeric.BinarySegment(white, black, 1),
                          screen,
                          vt)

    # Create a plot data object and give it this data
    pd = ArrayPlotData()
    pd.set_data("imagedata", image)

    # Create the plot
    plot = Plot(pd)
    img_plot = plot.img_plot("imagedata")[0]

    # Tweak some of the plot properties
    plot.title = "Abstract Rendering"
    plot.padding = 50

    return plot


#===============================================================================
# Attributes to use for the plot view.
size=(800,600)
title="Basic Colormapped Image Plot"

#===============================================================================
# # Demo class that is used by the demo.py application.
#===============================================================================
class Demo(HasTraits):
    plot = Instance(Component)

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(size=size),
                             show_label=False),
                        orientation = "vertical"),
                    resizable=True, title=title
                    )
    def _plot_default(self):
         return _create_plot_component()

demo = Demo()

if __name__ == "__main__":
    demo.configure_traits()
