from setuptools import setup


common_deps = ['numpy', 'torch', 'gym', 'nle', 'minihack']

d3qn_deps = ['ray[air]']

ppo_deps = ['pygame', 'stable_baselines3', 'tensorboard']

deps = common_deps + d3qn_deps + ppo_deps

setup(
    name="minihack-rl",
    version="0.0.1",
    description="Assignment for COMS4061A",
    url='git@github.com:mamello-justice/minihack-rl.git',
    author="Nilesh Jain, Mamello Seboholi, Abdel Njupoun, Thabo Rachidi",
    author_email="{2615122,1851317,2631227,1632496}@wits.students.ac.za",
    packages=['rlhack'],
    install_requires=deps
)
