from setuptools import setup, find_packages


with open("requirements.txt") as fh:
    requirements = [line.strip() for line in fh.readlines()]


setup(
    python_requires=">=3.9",
    install_requires=requirements,
    package_data={"mighty": ["requirements.txt"]},
    packages=find_packages(exclude=['test', 'examples', 'docs', 'checkpointing_test']),
    author="TODO",
    version="0.0.1",
    test_suite="nose.collector",
    tests_require=["mock", "nose"],
    long_description_content_type="text/markdown",
)
