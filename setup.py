import setuptools

with open("README_package.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spec_augment",
    version="0.0.3",
    author="MTrofficus",
    author_email="miguel.otero.pedrido.1993@gmail.com",
    description="Tensorflow Layer that implements the SpecAugment technique",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelisTrofficus/spec_augment",
    py_modules=["spec_augment"],
    package_dir={"": "spec_augment"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "tensorflow"
    ],
    python_requires='>=3.5',
)