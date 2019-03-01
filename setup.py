import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='ndf',
    version='0.1',
    author="Primoz Godec",
    author_email="primoz492@gmail.com",
    description="NumPy based deep learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/primozgodec/ndf",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "Pillow"],
    classifiers=[
     "Programming Language :: Python :: 3",
     "Operating System :: OS Independent",
    ]
)