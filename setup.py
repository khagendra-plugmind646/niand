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
        'cvxpy>=1.2.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Near-Field Integrated Sensing and Communications (NF-ISAC) Implementation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nf-isac",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
