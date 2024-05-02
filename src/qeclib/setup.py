from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

with open("_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "ipykernel",
    "pydantic",
    "numpy",
]

setup(
    name="QEClib",
    python_requires=">=3.10",
    version=version,
    packages=find_packages(),
    author="Jonathan Knoll",
    entry_points={},
    url="",
    install_requires=requirements,
    license="",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="",
)
