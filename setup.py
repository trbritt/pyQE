from setuptools import find_packages, setup

setup(
    name="pyQE",
    version="1",
    author="Tristan Britt",
    author_email="tristan.britt@mail.mcgill.ca",
    packages=find_packages(),
    install_requires=[
        "python=3.9.6",
        "numpy=1.21.1",
        "crystals=1.4.0",
        "npstreams=1.6.6",
        "scipy=1.7.0",
        "scikit-ued=2.1.5",
        "spglib=1.16.1",
        "pathlib",
        "tqdm=4.61.2",
        "pyqt5-sip=4.19.18",
        "qdarkstyle=2.8.1",
        "vispy=0.7.3",
        "imageio=2.9.0",
        "matplotlib-base=3.4.2",
    ]
)