from setuptools import setup, find_packages

setup(
    name="da",
    version="0.1.0",
    packages=find_packages(),
    description="Conditionally Gaussian Data Assimilation methods and experiments",
    author="Lydia Tolman",
    install_requires=["numpy","scipy","matplotlib"],
    python_requires=">=3.10",
)
