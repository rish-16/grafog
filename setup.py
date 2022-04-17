from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup(
  name = 'grafog',
  packages = find_packages(exclude=[]),
  version = '0.1',
  license='MIT',
  description = 'Graph Data Augmentations for PyTorch Geometric',
  long_description_content_type="text/markdown",
  long_description=README,
  author = 'Rishabh Anand',
  author_email = 'mail.rishabh.anand@gmail.com',
  url = 'https://github.com/rish-16/grafog',
  keywords = [
    'machine learning',
    'graph deep learning',
    'data augmentations'
  ],
  install_requires=[
    'torch>=1.10',
    'torch_geometric>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
