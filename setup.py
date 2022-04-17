import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="my_package-20CS30036",
    version="0.0.1",
    author="Yashwant Krishna Pagoti",
    author_email="pagotiyashwantkrishna@gmail.com",
    description="A package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
