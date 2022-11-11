from distutils.core import setup

deps = ['nle', 'minihack', 'ray[air]', 'gym', 'torch', 'numpy']

setup(
    name="minihack-rl",
    version="0.0.1",
    description="Assignment for COMS4061A",
    url='git@github.com:mamello-justice/minihack-rl.git',
    author="Mamello Seboholi",
    author_email="1851317@wits.students.ac.za",
    packages=['rlhack'],
    install_requires=deps
)
