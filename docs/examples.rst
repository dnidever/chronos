********
Examples
********


Running chronos
===============
The simplest way to run |Chronos| is with the command-line script ``chronos``.  The only required argument is the name of an image FITS file.

.. code-block:: bash

    chronos image.fits

By default, |Chronos| doesn't print anything to the screen.  So let's set the ``--verbose`` or ``-v`` parameter.


Running Chronos from python
==============================
|Chronos| has multiple modules.

    >>> import chronos
    >>> image = chronos.read('image.fits')
