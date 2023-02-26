import os
import re
from setuptools import setup


dir_path = os.path.dirname(os.path.realpath(__file__))
init_string = open(os.path.join(dir_path, 'frankenz4DESI', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)


setup(
    name='QFA',
    url='https://github.com/ZechangSun/QFA',
    version=__version__,
    author='Zechang Sun',
    author_email='szc22@mails.tsinghua.edu.cn',
    packages=["frankenz4DESI"],
    license="MIT",
    description="wrapper for frankenz",
    long_description=open("README.md").read(),
    package_data={"": ["REAMDME.md", "LICENSE", "AUTHORS.md"]},
    include_package_data=True,
    install_requires=["numpy", "torch", "yacs", "typing", "scipy", "h5py"],
    keywords=["photoz", "posterior distribution", "Bayesian analysis"],
    classifiers=["Development Status :: 3 - Alpha",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: English",
                 "Programming Language :: Python :: 3.10",
                 "Operating System :: OS Independent",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 "Intended Audience :: Science/Research"]
)