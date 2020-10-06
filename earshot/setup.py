#!/usr/bin/env python

from distutils.core import setup,Command

# this bit will go in here if we want to use python setup.py test
'''
class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable,'tests/runtests.py'])
        raise SystemExit(errno)
'''

# need to get all the dependent packages in properly at some point
setup(name='earshot',
      version='0.1.0',
      description='A minimal neural network model of incremental human speech recognition',
      author='various',
      author_email='various',
      url='path-to-earshot-repo',
      packages=['earshot'],
      package_dir = {'earshot': ''},
      include_package_data=True,
      #cmdclass = {'test': PyTest},
      license='MIT',
      classifiers=[
          'License :: OSI Approved :: BSD-3 License',
          'Intended Audience :: Developers',
          'Intended Audience :: Scientists',
          'Programming Language :: Python',
          'Topic :: ??',
      ],
    )
