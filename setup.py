from setuptools import find_packages, setup

setup(
    name='ionmodecalc',
    packages=find_packages(include = ['ionmodecalc']),
    version='0.9.1',
    description='Python library for calculation of motional mode spectrum of ion strings',
    author='Mikhail Popov',
    license='MIT',
    install_requires=['numpy', 'matplotlib', 'scipy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)