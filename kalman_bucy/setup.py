from setuptools import setup, find_packages

setup(
    name="kalman_bucy_filter",
    version="0.1.0",
    packages=find_packages(),
    description="Simple Kalman-Bucy filter for conditionally Gaussian systems",
    author="Lydia Tolman",
    install_requires=["numpy"],
    python_requires=">=3.9",
)
