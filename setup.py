from setuptools import setup
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='deepthink',
    version='0.1.5',
    description='Deep Learning library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Caff1982/DeepThink',
    author='Stephen Cafferty',
    author_email='stephencafferty@hotmail.com',
    license='MIT',
    packages=['deepthink'],
    install_requires=['numpy>=1.22.3',
                      'matplotlib>=3.6.2',
                      'scikit-learn>=1.1.2',
                      'tqdm>=4.64.1',
                      'pandas>=1.4.1'
                      ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
)
