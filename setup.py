from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['tensorflow==2.3',
                     'tensorflow-probability==0.11.0',
                     'tensorflow_datasets',
                     'torch==1.4',
                     'torchvision==0.5.0',
                     'numpy',
                     'nltk',
                     'networkx',
                     'dm-tree',
                     'matplotlib',
                     'dataclasses',
                     'mosestokenizer',
                     'subword-nmt',
                     'vizseq']


PACKAGES = [package
            for package in find_packages() if
            package.startswith('voi')]


setup(name='voi',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=PACKAGES,
      description='Variational Order Inference')
