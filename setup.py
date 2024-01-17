# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="seamless_communication",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"": ["py.typed", "cards/*.yaml"]},
    description="AIME fork of SeamlessM4T -- Massively Multilingual & Multimodal Machine Translation Model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="AIME GmbH / Fundamental AI Research (FAIR) at Meta",
    url="https://github.com/aime-labs/seamless_communication",
    license="Creative Commons"
)
