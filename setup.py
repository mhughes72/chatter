from setuptools import setup

setup(
    name='topicxtract_api_full',
    version='0.2',
    package_data = {'topicxtract_api': ['DATA/*.pkl', 'DATA/*.hdf5']},
    packages=['topicxtract_api', 'topicxtract_api.ProcessorChain', 'topicxtract_api.util', 'topicxtract_api.topicxtract_structs'],
    license='',
    long_description=open('README.md').read(),
)