## Overview
This repository contains implementations of Reinforcement Learning (RL) agents, including DQN, SAC, and PPO, organized across the main, sac_env, and PPO branches.
Each agent is integrated with its corresponding state representation and reward shaping, both defined in environment.py.

## Setup

### Python vesion to be used
* Python 3.11.x  (tested with 3.11.9)


### 1. Clone the Repository

Clone the repository using the following command:
- git clone https://github.com/fivecodingfive/ducks-riding-lamas.git
- cd ducks-riding-lamas


### 2. Add the Dataset

Copy the dataset from Moodle.

Drop the data/ folder—containing variant_0/, variant_1/, and variant_2/—into the root directory of your local clone.

Your project structure should then look like this:
- ducks-riding-lamas/
	- data/
	- greedy.py
	- environment.py
	- ...


### 3. Create and Activate the Virtual Environment (inside the repo)

Windows
- python -m venv venv
- .\venv\Scripts\Activate.ps1

MacOS / Linux
- python3 -m venv venv
- source venv/bin/activate


### 4. Install the exact package versions
- python -m pip install -r requirements.txt
