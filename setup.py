from setuptools import setup, find_packages

setup(
    name="smart-sell-advisor",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "flask==2.3.3",
        "flask-cors==4.0.0",
        "numpy==1.26.4",
        "pandas==2.1.4",
        "scikit-learn==1.3.2",
        "scipy==1.11.4",
        "joblib==1.3.2",
        "python-dotenv==1.0.0",
        "gunicorn==21.2.0",
    ],
    python_requires=">=3.10,<3.12",
)
