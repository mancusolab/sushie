import setuptools as st

with open("README.md", "r") as fh:
    long_description = fh.read()


st.setup(
    name="sushie",
    version="0.1.0",
    author="Zeyun Lu, Nicholas Mancuso",
    author_email="zeyunlu@usc.edu, nicholas.mancuso@med.usc.com",
    description="Sum of Single Shared effects among multiple ancestries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mancusolab/sushie",
    packages=st.find_packages(),
    package_data={"sushie": ['data/ld_blocks/*.bed']},
    install_requires=[
        # this is the minimum to perform fine-mapping given a prebuilt db.
        # functions that require addtl modules will warn/raise error
        # this is to minimize dependencies for the most-used scenario
        "numpy",
        "scipy",
        "pandas>=0.23.0",
        "pandas-plink",
        # limix
      ],
    scripts=[
        "bin/sushie",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
