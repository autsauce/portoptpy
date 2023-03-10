import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="portfolio-optimizer-python",
    version="0.1.0",
    author="Maddox Southard",
    author_email="maddoxsouthard@yahoo.com",
    description="A Python library for interfacing with the Portfolio Optimizer API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autsauce/portfolio-optimizer-python",
    packages=setuptools.find_packages(),
    install_requires=[
        "yfinance",
        "requests",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)