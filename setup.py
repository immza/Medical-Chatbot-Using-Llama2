from setuptools import find_packages, setup

setup(
    name='Medical Chatbot',
    version= '0.0.0',
    author='MZ Ayan',
    author_email='moinuddinzubair26@gmail.com',
    packages=find_packages(), #will look for the constructor files
    #it makes src useable as a package
    install_requires=[]
)