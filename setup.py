from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

# get __version__ from _version.py
ver_file = path.join('tensorpi', 'version.py')
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.rst
def readme():
    with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
        return f.read()


# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='tensorpi',
    version=__version__,
    description='Python library for low-rank tensor learning',
    long_description=readme(),
    long_description_content_type='text/x-rst',
    author='Xinyu Chen',
    author_email='chenxy346@gmail.com',
    url='https://github.com/xinychen/tensorpi',
    download_url='https://github.com/xinychen/tensorpi/archive/master.zip',
    keywords=['tensor learning', 'tensor computation', 'matrix completion',
              'tensor completion'],
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=['setuptools>=38.6.0'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)