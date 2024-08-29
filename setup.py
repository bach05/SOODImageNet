from setuptools import setup, find_packages

setup(
    name="SOODImageNet",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add your project dependencies here, as listed in requirements.txt
        # e.g., 'numpy', 'torch', 'Pillow'
        'torch', 'torchvision', 'numpy', 'matplotlib', 'opencv-python', 'pyyaml', 'tqdm'
    ],
    entry_points={
        'console_scripts': [
            # You can define command-line scripts here if necessary
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A description of your SOODImageNet package",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/SOODImageNetDataset",  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
