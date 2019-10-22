from distutils.core import setup
import os

setup(
    name='glia',
    version='',
    author='Laurent Lejeune',
    author_email='laurent.lejeune@artorg.unibe.ch',
    description='glia',
    packages=['glia'],
    package_dir={'glia': '/home/krakapwa/Documents/software/glia/build/src'},
    package_data = {'glia': ['libglia.so']}
)
