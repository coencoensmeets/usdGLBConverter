#!/usr/bin/env python3
"""
Setup script for RobotUSD - USD to glTF Converter for Robotics Applications
"""

import os
import sys
from setuptools import setup, find_packages

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "USD to glTF Converter for Robotics Applications"

# Read requirements from requirements.txt if it exists
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Get version from the package
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'src', 'robotusd', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"

# Check Python version
if sys.version_info < (3, 7):
    sys.exit('Python 3.7 or later is required.')

# Core requirements
if not (sys.version_info >= (3, 8) and sys.version_info < (3, 13)):
    sys.exit('Python >=3.8 and <3.13 is required for RobotUSD and usd-core. Please install a compatible Python version. Current python version: {}.{}.{}'.format(
        sys.version_info.major, sys.version_info.minor, sys.version_info.micro))

install_requires = [
    'numpy>=1.19.0',
    # usd-core is only available for Python >=3.8 and <3.13
    'usd-core>=23.05; python_version>="3.8" and python_version<"3.13"',
]

# Optional requirements for different use cases
extras_require = {
    'dev': [
        'pytest>=6.0',
        'pytest-cov>=2.0',
        'black>=21.0',
        'flake8>=3.8',
        'mypy>=0.910',
    ],
    'docs': [
        'sphinx>=4.0',
        'sphinx-rtd-theme>=0.5',
        'myst-parser>=0.15',
    ],
    'performance': [
        'numba>=0.53',  # For JIT compilation of math operations
    ],
    'visualization': [
        'matplotlib>=3.3',
        'plotly>=5.0',
    ]
}

# Add development requirements to test
extras_require['test'] = extras_require['dev']

# All optional dependencies
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name='robotusd',
    version=get_version(),
    author='Vectioneer',
    author_email='support@vectioneer.com',
    description='USD to glTF Converter for Robotics Applications',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    
    # Package configuration
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    
    # Dependencies
    python_requires='>=3.7',
    install_requires=install_requires + read_requirements(),
    extras_require=extras_require,
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'robotusd-convert=robotusd.cli:main',
            'robotusd-info=robotusd.cli:info_command',
        ],
    },
    
    # Package data
    package_data={
        'robotusd': [
            'examples/*.py',
            'examples/*.usd',
            'templates/*.json',
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Multimedia :: Graphics :: 3D Modeling',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Environment :: Console',
    ],
    
    # Keywords for discoverability
    keywords=[
        'usd', 'gltf', 'robotics', 'converter', '3d', 'graphics',
        'omniverse', 'pixar', 'universal scene description',
        'robot', 'simulation', 'mesh', 'materials'
    ],
    
    # Additional metadata
    license='MIT',
    platforms=['any'],
    zip_safe=False,  # USD files are often large and better extracted
    
    # Custom commands
    cmdclass={},
    
    # Test configuration
    test_suite='tests',
    tests_require=extras_require['test'],
)

# Only show message during actual installation, not during build
if 'install' in sys.argv:
    import atexit
    atexit.register(print_post_install_message)