from setuptools import setup, find_packages

with open ("readme.md", "r", encoding="utf-8") as fh:
    long_des = fh.read()

setup(
    name = "bgvODESlovr",
    version = "0.0.1",
    author = "bgvmysore",
    description = 'This is a containing ODE solvers namely, RK4, EulerMethods',
    long_description = long_des,
    Liscense = 'MIT',
    packages = find_packages(),
    install_requires = ['tqdm', 'numpy'],
    python_requires = '>=3.8'
)
