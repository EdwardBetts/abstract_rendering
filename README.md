Abstract Rendering
======
[![Build Status](https://travis-ci.org/ContinuumIO/abstract_rendering.svg?branch=master)](https://travis-ci.org/ContinuumIO/abstract_rendering)

Information visualization rests on the idea that a meaningful relationship
can be drawn between pixels and data.  This is most often mediated by
geometric entities (such as circles, squares and text) but always involves
pixels eventually to display.  In most systems, the pixels are tucked away
under levels of abstraction in the rendering system.  Abstract Rendering
takes the opposite approach: expose the pixels and gain powerful pixel-level
control.  This pixel-level power is a complement to many existing visualization
techniques.  It is an elaboration on rendering, not an analytic or projection step,
so it can be used as epilogue to many existing techniques.


In standard rendering, geometric objects are projected to an image and 
represented on that image's discrete pixels.  The source space is a
canvas that contains logically continuous geometric primitives 
and the target space is an image that contains discrete colors.
Abstract Rendering fits between these two states.  It introduces
a discretization of the data at the pixel-level, but not necessarily all
the way to colors.  This enables many pixel-level concerns to be efficiently 
and concisely captured.

This repository is a Python implementation of the abstract rendering framework.
For more details on the conceptual framework and examples of applications,
please see the [more general repository](https://github.com/JosephCottam/AbstractRendering/).
Sample images can be found in [the central wiki](https://github.com/JosephCottam/AbstractRendering/wiki).
