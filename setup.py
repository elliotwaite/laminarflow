import setuptools

with open('README.md') as f:
  long_description = f.read()

setuptools.setup(
  name='laminarflow',
  version='0.0.7',
  description='Streamline your TensorFlow workflow.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/elliotwaite/laminarflow',
  author='Elliot Waite',
  author_email='elliot@elliotwaite.com',
  packages=setuptools.find_packages(),
  install_requires=[
    'numpy >= 1.13.3',
    'tensorflow >= 1.7.0',
  ],
  classifiers=(
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
  ),
  license='Apache 2.0',
  keywords='laminarflow laminar tensorflow tensor tfrecord tfrecords ml '
           'machine learning ai artificial intelligence deep neural network '
           'networks',
)
