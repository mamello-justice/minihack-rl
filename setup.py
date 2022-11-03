from distutils.core import setup

deps = ['nle', 'minihack', 'gym', 'torch', 'numpy']

setup(
    name="minihack_rl",
    version="0.0.1",
    description="Assignment for COMS4061A",
    url='git@github.com:mamello-justice/minihack_rl.git',
    author="Mamello Seboholi",
    author_email="1851317@wits.students.ac.za",
    packages=['minihack_rl'],
    install_requires=deps
)
