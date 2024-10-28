from setuptools import setup, find_packages

setup(
    name="Automatic_Feature_Engineering",  # Unique name for PyPI
    version="0.3.0",  # Start with a version number
    author="Sagar Sambhwani",
    author_email="sagar.2001.a20@gmail.com",
    description="A package for automated feature engineering with support for categorical and numerical data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sagarsambhwani/AutoFeatureEngineering",  # Link to your GitHub repo
    packages=find_packages(),  # Automatically find and include the package
    install_requires=[
        "pandas",
        "scikit-learn",
        "category_encoders",
        "numpy",
        "tsfresh"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
