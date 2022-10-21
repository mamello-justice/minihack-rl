# MiniHack (Reinforcement Learning Assignment)

## Getting Started

### Requirements

- Linux/Unix OS or Docker (Recommended)

### Develop

1. Clone repository `git clone https://`
2. Create a wip/development branch
3. Push branch to Github
4. Create Pull Request (PR) and ping in group on discord for review
5. Merge PR to main branch (Please note that main branch is `protected`)

## Code Structure

- [.vscode](./vscode) - Visual Studio Code configuration
- [config](./config) - Don't place anything in here, used to mount docker configuration for webtop image
- [docs](./docs) - Documentation along with assignment questionnaire/instructions
- [minihack](./minihack) - Main project code
  - [dqn](./minihack/dqn) - Deep Q-Network family code (value-function based methods)
  - [environments](./minihack/environments) - Environments for the project
  - [mcts](./minihack/mcts) - Monte-Carlo Tree Search code (model-based method)
  - [option-critic](./minihack/option-critic) - Option Critic code (hierarchical method)
  - [reinforce](./minihack/reinforce) - REINFORCE code (policy method)
- [papers](./papers) - Papers used for the project (NOT SURE IF THIS IS LEGAL BUT I GUESS IF WE KEEP IT PRIVATE THEN NO ISSUE???)
- [test](./test) - Good old testing
- [docker-compose](./docker-compose.yml) and [Dockerfile](./Dockerfile) - Docker config files (IF USING DOCKER/WINDOWS)
- [formulae](./formulae.tex) - Latex formulae for the different methods (cheatsheet)
- [references](./references.bib) - Paper references
- [setup](./setup.py) - Python module setup file

## Todo

- Setup a CI/CD to build and package module on Github (Private) for easy access in Notebooks
- Notebooks for graphics (charts/tables etc.)
- Package requirements/dependencies in setup.py
- Unit testing
