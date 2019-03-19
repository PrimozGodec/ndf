import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='ndf',
    version='0.1.2',
    author="Primoz Godec",
    author_email="primoz492@gmail.com",
    description="NumPy based deep learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/primozgodec/ndf",
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.13.0", "Pillow"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    test_suite='ndf.tests.suite'
)