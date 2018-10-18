=========================
Introduction to *kungpao*
=========================

Goals
-----

*kungpao* is a collection of programming tools which--along with Python's extensive numerical and scientificl libraries--permit control and characterization of an adaptive optics (AO) system. It was largely inspired by ALPAO's `ACE <http://www.alpao.fr/products_ACE.html>`_, and provides very similar functionality using open source software.

Quickstart
----------




Architecture
------------

Object model
************

The object model is based on the fundamental components and concepts of an AO system: a **camera** object responsible for acquiring raw images from a CCD or CMOS sensor; a **sensor** object responsible for detecting and centroiding spots and computing wavefront slopes; a **reconstructor** object responsible for reconstructing the wavefront from slope data; and a deformable **mirror** object resposible for controlling a deformable mirror (or other phase modulator).

The **camera** and **reconstructor** are properties of (encapsulated by) the **sensor** object. The **sensor** and **mirror** are properties of the **loop** object.

Some classes can be instantiated and used individually. For example, a **camera** can be used to acquire and process spots images. A **mirror** can be used to shape the imaging system's wavefront, without using a wavefront sensor (in the case of sensorless AO or image sharpening, for instance). Similarly, a **sensor** object can be instantiated to measure the system's wave aberrations without necessarily correcting wave errors.

The canonical mode of operation is instantiating a **sensor** (with **camera** and **reconstructor** properties) and a **mirror**, and then instantiating a **loop** object with these two:

A minimal AO loop can be run as shown in the following script (``/examples/example_minimal_loop.py``):

.. literalinclude:: ../../examples/example_minimal_loop.py

Tools
*****

PyAO makes use of a number of existing Python libraries. These are listed below.

ctypes
  `ctypes <http://docs.python.org/2/library/ctypes.html>`_ is currently part of the Python standard library. It facilitates the access
  of 'foreign functions' and associated data types. In PyAO, ctypes is used to expose libraries of camera and deformable mirror functions,
  generally compiled in the devices' DLLs.

NumPy and SciPy
  `NumPy <http://numpy.scipy.org/>`_ and `SciPy <http://scipy.org/>`_ provide optimized numerical computing functions (e.g. matrix algebra,
  Fourier transformations, etc.) used to perform reconstruction and AO control. Together these packages provide scientific computing
  capabilities similar to MATLAB.

PyOpenGL and glumpy
  `PyOpenGL <http://pyopengl.sourceforge.net/>`_ provides Python bindings to OpenGL, and `glumpy <http://code.google.com/p/glumpy/>`_
  nicely unifies PyOpenGL textures with NumPy arrays, which permits rapid, GPU-accelerated visualization of various AO matrices. This is
  useful mainly for running the system in closed loop, where allocation of system resources such as CPU and RAM to visualization comes
  at the cost of AO computation frame rate.

matplotlib
  `matplotlib <http://matplotlib.org/>`_ provides a MATLAB-like set of tools for exploratory data analysis and visualization.

glumpy and matplotlib are complementary packages, useful for visualizing data. glumpy excels at high-speed visualization, appropriate for a real time application, while matplotlib is better at producing static visualizations and publication-quality graphics.
