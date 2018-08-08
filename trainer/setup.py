
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='{name_of_model}',
    version='0.1',
    author = '{name of author}',
    author_email = '{email@example.com}',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='{Some description}',
    requires=[]
)