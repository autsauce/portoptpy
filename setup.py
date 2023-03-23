import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="portoptpy",
    version='0.1.6',
    author="Maddox Southard",
    author_email="maddoxsouthard@yahoo.com",
    description="A Python library for interfacing with the Portfolio Optimizer API: https://docs.portfoliooptimizer.io/",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autsauce/portoptpy",
    packages=setuptools.find_packages(),
    install_requires=[
            'yfinance',
            'requests',
            'aiohttp',
            'asyncio',
            'pandas',
            'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
