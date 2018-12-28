#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "potara",
    version = "1.0.1",
    author = "Remi Bois",
    author_email = "remibois.contact@gmail.com",
    description = ("A multi-document summarizer based on ILP and sentence fusion."),
    license = "Apache",
    keywords = "summarization",
    packages = ['potara', 'tests'],
    test_suite =  "tests.test_all",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url='https://github.com/sildar/potara',
    classifiers=['Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Development Status :: 5 - Production/Stable',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Information Analysis',
                 'Topic :: Text Processing :: Linguistic',
                ],
    install_requires=['gensim==3.6.0',
                      'networkx==1.8.1',
                      'nltk==3.4',
                      'pulp',
                      'setuptools==40.6.3',
                      'python-coveralls==2.4.2',
                      'six==1.12.0',
                      'requests>=2.21.0',
                      ]
)