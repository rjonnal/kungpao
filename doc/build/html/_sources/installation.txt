====================
Installation of PyAO
====================

Installing Python and other packages
------------------------------------

Installation of PyAO consists largely of installing Python and some
extensions. Copies of the installers should ordinarily be distributed
with PyAO, to ensure compatibility among packages. The extensions
should be installed in the order below, to ensure that missing
dependencies do not cause installations to abort.

Exact version numbers are provided below, and should be used when
possible. Version numbers in Python have the format M.m.R (for Major,
minor, and Release, respectively). Slight differences between the
versions listed below and your installed versions are unlikely to
prevent PyAO from working; for example, PyAO runs on Python 2.6.* and
2.7.*, but does not run on any minor version or revision of Python 3.

    1. `Python 2.7.3
    <http://www.python.org/ftp/python/2.7.3/python-2.7.3.msi>`_
    
    2. `NumPy 1.6.2
    <http://pypi.python.org/packages/2.7/n/numpy/numpy-1.6.2.win32-py2.7.exe>`_
    
    3. `SciPy 0.11.0
    <http://sourceforge.net/projects/scipy/files/scipy/0.11.0/scipy-0.11.0-win32-superpack-python2.7.exe>`_
    
    4. `Matplotlib 1.1.1
    <http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.1.1/matplotlib-1.1.1.win32-py2.7.exe>`_
    
    5. `PyOpenGL 3.0.1
    <http://pypi.python.org/packages/any/P/PyOpenGL/PyOpenGL-3.0.1.win32.exe>`_
    
    6. `Glumpy 0.2.1
    <http://code.google.com/p/glumpy/downloads/detail?name=glumpy-0.2.1.tar.gz>`_

By default, Python will be installed in ``C:\Python27``. This must be
appended to the PATH environment variable before proceeding.

The above links point at .msi or .exe installers, with the exception
of Glumpy. To install Glumpy, unpack the .tar.gz file, open a command
window and navigate to the directory containing setup.py and issue the
command ``python setup.py install``.

Installing PyAO
---------------

When unpacked, PyAO results in the following directory structure:

    * ``pyao/pyao``: The core AO modules and classes, corresponding to,
      e.g., cameras, wavefront sensors, mirrors, etc.

    * ``pyao/pyao_doc``: The documentation source and HTML and LaTeX
      builds; see `Sphinx <http://sphinx-doc.org/>`_ documentation for
      details.

    * ``pyao/pyao_gui``: GUIs for various ways of running the system, e.g. closed loop, building a
      poke matrix, etc.

    * ``pyao/pyao_scripts``: Other Python scripts used for system characterization and testing; examples
      of non-standard use of the system, e.g. temporal bandwidth calculation, etc.

    * ``pyao_etc.tar.gz`` (or ``pyao_etc.zip``): An archive containing the pyao_etc folder. This should be
      unzipped to a separate location (specified by the environment variable ``PYAOETCPATH``, see below). It contains
      installation-specific components, viz. those pertaining to a particular instance of the application (e.g. a
      single AO system or a single simulator). One rule of thumb about
      this directory: don't copy someone else's onto your system, or
      your reference coordinates, influence functions, etc. will be
      clobbered. This folder should contain:

        * ``pyao_etc/config/ctrl``: Influence function(s) (poke matrices).

        * ``pyao_etc/config/dcf``: Camera configuration file(s).

        * ``pyao_etc/config/dm``: Mirror flat file(s).

        * ``pyao_etc/config/ref``: Reference coordinates for the Shack-Hartmann wavefront sensor.

        * ``pyao_etc/data``: Other (non-configuration) data the system might use, such as spots
          images used to run the system in simulation mode.

        * ``pyao_etc/log``: Where the log files are saved.

Three environmental variable must be set: 
    1. ``PYTHONPATH`` must be set to the full path to the top level ``pyao`` directory, e.g. ``C:\Program Files\pyao``.
    2. ``PYAOPATH`` must be set to the top level ``pyao`` directory as well, e.g. ``C:\Program Files\pyao``.
    3. ``PYAOETCPATH`` must be set to the location of the ``pyao_etc`` directory. This directory should be on the local machine, whereas the top level ``pyao`` directory can reside anywhere (e.g. network share, Dropbox, etc.). ``pyao_etc`` should be local in order to optimize data logging (which is faster on a local drive) and also to prevent collisions between 1) system-specific parameters set in ``pyao_config.py``, which resides ``pyao_etc`` or 2) system-specific configuration files, which reside in ``pyao_etc/config/ctrl``, ``pyao_etc/config/dcf``, ``pyao_etc/config/dm``, and ``pyao_etc/config/ref``.

In Windows these are set by right-clicking on 'My Computer', then 'Properties'->'Advanced'->'Environment Variables', and then editing them by hand. 

In linux, you may append the following to your .bashrc, .bash_profile, or equivalent startup script, replacing dummy paths with your actual paths:

 .. code-block:: bash
 
    export PYTHONPATH=$PYTHONPATH:/path/to/pyao
    export PYAOPATH=/path/to/pyao
    export PYAOETCPATH=/path/to/pyao_etc

Please note the difference between the PYTHONPATH definition and the PYAOPATH and PYAOETCPATH definitions. The first appends to any existing definition while the others redefine. This is an important difference; the last two are environmental variables used only by pyao, which expects to find a single path and not a series of paths. PYTHONPATH by contrast may be used by any number of other python applications on your system, necessitating appending. 

The redundancy between 1 and 2 is due to an architectural error that will hopefully be fixed in the next version.
