from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    This function is used to return list of requirements
    """
    requirement_list:List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!='-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("Error: requirements.txt file not found.")

    """
    Write a code to read requirements.txt file and append each requirements in requirement_list variable.
    """
    return requirement_list

setup(
    name = "Surname & Caste Prediction",
    version = "0.0.1",
    author = "Arpit Shourya",
    author_email = "arpitshourya2233@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
