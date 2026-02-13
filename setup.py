from setuptools import setup, find_packages

setup(
    name="smart-sell-advisor",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "flask>=3.1.0",
        "flask-cors>=5.0.0",
        "numpy>=2.0.1",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.0",
        "scipy>=1.14.1",
        "joblib>=1.4.2",
        "python-dotenv>=1.0.1",
        "gunicorn>=23.0.0",
    ],
    python_requires=">=3.11",
)