# MiniHack (Reinforcement Learning Assignment)

## Getting Started

### Requirements

- Linux/Unix OS or Docker (Recommended)

### Develop

1. Clone repository

   ```bash
   git clone https://github.com/mamello-justice/minihack-rl.git
   ```

2. Create a wip/development branch
3. Push branch to Github
4. Create Pull Request (PR) and ping in group on discord for review
5. Merge PR to main branch (Please note that main branch is `protected`)

### Training

> NB: Before training, make sure NLE and Minihack are install

```bash
# Install CMake & Other requirements
apt update -qq && apt install -qq -y flex bison libbz2-dev libglib2.0-0 libsm6 libxext6 cmake

# Install NLE
cd ./third_party/nle
pip install .[all]

# Install Minihack
cd ./third_party/minihack
# Fix issues like env -> nethack in base.py
pip install .
```

**1. D3QN**

```bash
# Install required dependencies
pip install .

python -m rlhack.d3qn.train
# OR
cd rlhack/d3qn
python train.py
```

**2. PPO**

```bash
# Install required dependencies
pip install .

python -m rlhack.ppo.train
# OR
cd rlhack/ppo
python train.py
```

## Code Structure

- [.vscode](./vscode) - Visual Studio Code configuration
- [config](./config) - Don't place anything in here, used to mount docker configuration for webtop image
- [docs](./docs) - Documentation along with assignment questionnaire/instructions
- [export](./export) - Exported figures
- [papers](./papers) - Papers used for the project (NOT SURE IF THIS IS LEGAL BUT I GUESS IF WE KEEP IT PRIVATE THEN NO ISSUE???)
- [rlhack](./rlhack) - Main project code
  - [d3qn](./rlhack/d3qn) - Deep Q-Network family code (value-function based methods)
  - [ppo](./rlhack/ppo) - Proximal Policy Optimization code (policy based)
- [tests](./tests) - Good old testing
- [third_party](./third_party) - Third party/vendor submodules
  - [minihack](./third_party/minihack/)
- [docker-compose.yml](./docker-compose.yml) and [Dockerfile](./Dockerfile) - Docker config files (IF USING DOCKER/WINDOWS)
- [formulae.tex](./formulae.tex) - Latex formulae for the different methods (cheatsheet)
- [references.bib](./references.bib) - Paper references
- [setup.py](./setup.py) - Python module setup file
