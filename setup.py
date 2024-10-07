from setuptools import setup, find_packages

setup(
    name='Image-Toolbox',
    version="0.1.0",
    description="A short description of your package",
    packages=find_packages(),  # Automatically find and include your package
    install_requires=[         # List of dependencies
      "SimpleITK",
      "ipywidgets",
      "matplotlib",
      "numpy",
      "opencv-python"
    ],
    classifiers=[              # Metadata
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
)
