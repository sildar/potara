#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "potara",
    version = "0.1.0",
    author = "Remi Bois",
    author_email = "remibois.contact@gmail.com",
    description = ("A multi-document summarizer based on ILP and sentence fusion."),
    license = "Apache",
    keywords = "summarization",
    packages = ['potara', 'tests'],
    test_suite =  "tests.test_all",
    long_description=read('README.md'),
    classifiers=[
    ],
)