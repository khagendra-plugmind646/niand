from setuptools import setup, find_packages

setup(
    name="nf_isac",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'joblib>=1.1.0',
        'cvxpy>=1.2.0',  # *** CORRECTION: Added missing dependency
    ],
    author="Zhaolin Wang et al.", # Citing original author from README
    description="Near-Field Integrated Sensing and Communications (NF-ISAC) Implementation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)