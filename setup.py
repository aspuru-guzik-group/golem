"""Golem: An Algorithm for Robust Experiment and Process Optimization
"""

import versioneer
from setuptools import setup, Extension


# readme file
def readme():
    with open('README.md') as f:
        return f.read()


# extensions
ext_modules = [Extension("golem.extensions", ["src/golem/extensions.c"])]

# -----
# Setup
# -----
setup(name='golem',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Golem: An Algorithm for Robust Experiment and Process Optimization',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
      ],
      url='https://github.com/aspuru-guzik-group/golem',
      author='Matteo Aldeghi',
      author_email='matteo.aldeghi@vectorinstitute.ai',
      license='MIT',
      packages=['golem'],
      package_dir={'': 'src'},
      zip_safe=False,
      tests_require=['pytest', 'deap'],
      install_requires=['numpy', 'scipy>=1.4', 'scikit-learn', 'pandas'],
      python_requires=">=3.7",
      ext_modules=ext_modules
      )
