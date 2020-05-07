"""Golem: A Probabilistic Approach to Optimization Under Uncertain Inputs

Some description here...
"""

from setuptools import setup, Extension
import versioneer


def readme():
    with open('README.md') as f:
        return f.read()


# -----
# Setup
# -----
setup(name='golem',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        #'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
      ],
      url='https://github.com/matteoaldeghi/golem',
      author='Matteo Aldeghi',
      author_email='matteo.aldeghi@vectorinstitute.ai',
      #license='GPL 3',
      packages=['golem'],
      package_dir={'': 'src'},
      #include_package_data=True,
      zip_safe=False,
      tests_require=['pytest', 'deap'],
      install_requires=['numpy', 'scipy>=1.4', 'scikit-learn', 'pandas', 'cython'],
      python_requires=">=3.7"
      )
