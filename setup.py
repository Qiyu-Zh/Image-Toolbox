from setuptools import setup, find_packages

setup(
    name='ImageTool',
    version="0.1.0",
    description="A short description of your package",
    packages=find_packages(),  # Automatically find and include your package
    install_requires=[         # List of dependencies
      "SimpleITK",
      "ipywidgets",
      "matplotlib==3.5.1",
      "numpy==1.25",
      "opencv-python",
      "scikit-image",
      "antspyx==0.4.2",
      "monai",
      "scipy"
    ],
    classifiers=[              # Metadata
        "Programming Language :: Python :: 3",
    ],
    python_requires'>=3.8',
)
