from setuptools import setup, find_packages

setup(
    name="phylogenetic_analysis",
    version="2.0.0",
    description="Phylogenetic Polymorphism Analysis Tool with Decision Tree Interface",
    author="Nomlindelo Mfuphi",
    author_email="nmfuphi@csir.co.za",
    packages=find_packages(),
    install_requires=[
        "ete3==3.1.3",
        "pandas==2.2.3",
        "numpy==2.1.2",
        "scikit-learn==1.5.2",
        "scipy==1.13.1",
        "statsmodels==0.14.4",
        "pyyaml==6.0.2"
    ],
    entry_points={
        "console_scripts": [
            "phylogenetic_analysis=phylogenetic_analysis:main"
        ]
    },
    include_package_data=True,
    package_data={
        "": ["data/*", "config.yml"]
    }
)