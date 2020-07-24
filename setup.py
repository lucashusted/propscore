# To create new distro: python setup.py sdist bdist_wheel
# To upload: twine upload dist/*

import io
import os
from setuptools import setup, find_packages

dir = os.path.dirname(__file__)

with io.open(os.path.join(dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='propscore',
    version='0.0.3',
    description='Program to Calculate Optimal Propensity Score',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lucashusted/propscore',
    author='Lucas Husted',
    author_email='lucas.f.husted@columbia.edu',
    license='MIT',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        ],
    install_requires=['statsmodels','pandas','scipy'],
    python_requires='>=3',
    packages=find_packages()
)
