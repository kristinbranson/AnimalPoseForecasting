from setuptools import setup, find_packages

setup(
    name='AnimalPoseForecasting',
    version='0.1.0',
    packages=find_packages(include=['flyllm', 'flyllm.*', 'apf', 'apf.*', 'experiments', 'experiments.*'])
)

# run `pip install -e .` to run this in editable mode
