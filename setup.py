import os, sys, glob, re
from setuptools import setup, find_packages

def _get_version():
    line = open('py/nearly_nmf/_version.py').readline().strip()
    m = re.match("__version__\s*=\s*'(.*)'", line)
    if m is None:
        print('ERROR: Unable to parse version from: {}'.format(line))
        version = 'unknown'
    else:
        version = m.groups()[0]

    return version

setup_keywords = dict(
    name='nearly_nmf',
    version=_get_version(),
    description='Algorithms for generating NMF templates from noisy data with negative values.',
    url='https://github.com/dylanagreen/nearly_nmf',
    author='Dylan Green',
    author_email='dylanag@uci.edu',
    license='BSD 3-Clause',
    packages=find_packages("py"),
    package_dir={"": "py"},
    install_requires=['numpy>=1.21.0'],
    zip_safe=False,
)

setup(**setup_keywords)