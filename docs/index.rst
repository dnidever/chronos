.. chronos documentation master file, created by
   sphinx-quickstart on Tue Feb 16 13:03:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*******
Chronos
*******

Introduction
============
|Chronos| [#f1]_ is software to automatically fit isochrones to cluster photometry.

.. toctree::
   :maxdepth: 1

   install
   gettingstarted
   modules
   

Description
===========
|Chronos| has a number of modules to perform photometry.

|Chronos| can be called from python directly or the command-line script `hofer` can be used.


Examples
========

.. toctree::
    :maxdepth: 1

    examples


prometheus
==========
Here are the various input arguments for command-line script `prometheus`::

  usage: prometheus [-h] [--outfile OUTFILE] [--figfile FIGFILE] [-d OUTDIR]
                    [-l] [-p] [-v] [-t]
                    files [files ...]

  Run Chronos on an image

  positional arguments:
    files                 Images FITS files or list

  optional arguments:
    -h, --help            show this help message and exit
    --outfile OUTFILE     Output filename
    --figfile FIGFILE     Figure filename
    -d OUTDIR, --outdir OUTDIR
                          Output directory
    -l, --list            Input is a list of FITS files
    -p, --plot            Save the plots
    -v, --verbose         Verbose output
    -t, --timestamp       Add timestamp to Verbose output

*****
Index
*****

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
    
.. rubric:: Footnotes

.. [#f1] In Greek mythology, `Chronos <https://en.wikipedia.org/wiki/Chronos>`_ is a Titan that brings fire from the heavens down to humans on earth.