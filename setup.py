import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    sys.exit('Sorry, Python < 3.8 is not supported.')

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="froggeolib",
    packages=[package for package in find_packages()],
    version="1.0",
    license="GPL",
    description="My Geospatial library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Frogmane",
    author_email="",
    url="https://github.com/xznhj8129/froggeolib",
    download_url="",
    keywords=['Geospatial', 'geojson','mgrs','geo'],
    install_requires=['geographiclib','mgrs','geojson'],
    classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python',
          'Framework :: Robot Framework :: Library',
          'Topic :: Education',
    ]
)