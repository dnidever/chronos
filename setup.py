#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

pypiname = 'thechronos'

setup(name="thechronos",
      version='1.0.15',
      description='Automatic isochrone fitting to photometric data',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/chronos',
      packages=find_packages(exclude=["tests"]),
#      scripts=['bin/dopfit','bin/dopjointfit','bin/doppler'],
      requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils','emcee','cornner'],
      include_package_data=True,
)
