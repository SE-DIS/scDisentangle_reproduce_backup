from setuptools import setup, find_packages 

#here = path.abspath(path.dirname(__file__))

setup(
    name='scdisentangle',

    version='0.1',

    description='scdisentangle',
    
    url='',

    author='labi',
    author_email='',

    test_suite="", 
    
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[],

    keywords='',

    packages=find_packages(exclude=[]),

    install_requires=[
        # "igraph==0.10.6",
        # "scanpy==1.9.4",
        # "leidenalg==0.10.1",
        # "torch==2.0.1",
        # "matplotlib==3.5.1",
        # "numpy==1.23.5",
        # "umap-learn==0.5.3",
        # "yaml==5.4.1"
    ],

    package_data={
        '': ['*.txt', '*.rst', '*.tar.gz'],
    },

    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },
)
