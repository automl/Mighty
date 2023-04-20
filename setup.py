#!/usr/bin/env python3
from setuptools import setup, find_packages


setup(
    package_data={'mighty': ['requirements.txt']},
    packages=find_packages(exclude=['tests', 'examples', 'docs', 'checkpointing_test'])
)
