#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
	name='full_body_global_planner_ros',
	version='0.0.0',
	packages=find_packages(where='src'),
	package_dir={'': 'src'},
	zip_safe=False,
)
