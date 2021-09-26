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


chronos
=======
Here are the various input arguments for command-line script `chronos`::

  usage: chronos [-h] [--outfile OUTFILE] [--figfile FIGFILE] [-d OUTDIR]
                 [-l] [-p] [-v] [-t]
                 files [files ...]

  Run Chronos on a catalog

  positional arguments:
    files                 Catalog FITS files or list

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

.. [#f1] `Chronos <https://en.wikipedia.org/wiki/Chronos>`_, is the personification of time in pre-Socratic philosophy and later literature.
