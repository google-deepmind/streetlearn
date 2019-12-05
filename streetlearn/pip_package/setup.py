# Copyright 2018 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Setup module for turning StreetLearn into a pip package.
Based on: https://github.com/google/nucleus/blob/master/nucleus/pip_package/setup.py
This should be invoked through build_pip_package.sh, rather than run directly.
"""
import fnmatch
import os

from setuptools import find_packages
from setuptools import setup

def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


def is_python_file(fn):
  return fn.endswith('.py') or fn.endswith('.pyc')

headers = list(find_files('*.h', 'streetlearn'))

matches = ['../' + x for x in find_files('*', 'external')
           if not is_python_file(x)]

so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

for path in so_lib_paths:
  matches.extend(['../' + x for x in find_files('*', path)
                  if not is_python_file(x)])

setup(
    name='streetlearn',
    version='0.1.0',
    description='A library to aid navigation research.',
    long_description=
    """
    TODO
    """,
    url='https://github.com/deepmind/streetlearn',
    author='The StreetLearn Team at DeepMind',
    author_email='streetlearn@google.com',
    license='Apache 2.0',

    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: POSIX :: Linux',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='navigation tensorflow machine learning',
    packages=find_packages(exclude=['g3doc', 'testdata']),
    install_requires=['six', 'absl-py', 'inflection', 'wrapt', 'numpy',
                      'dm-sonnet', 'tensorflow', 'tensorflow-probability'],
    headers=headers,
    include_package_data=True,
    package_data={'streetlearn': matches},
    data_files=[],
    entry_points={},
    project_urls={
        'Source': 'https://github.com/deepmind/streetlearn',
        'Bug Reports': 'https://github.com/deepmind/streetlearn/issues',
    },
    zip_safe=False,
)
