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


### 3. Install Pixi

Follow the installation instructions at [pixi.sh](https://pixi.sh) for your platform (Windows, macOS, or Linux).


### 4. Install Dependencies and Activate Environment

Once pixi is installed, run:
- pixi install

This will create and activate a managed environment with all dependencies pinned to exact versions.


### 5. Run Tasks

Use pixi to run project tasks:
- pixi run train          # Run main training script
- pixi run test           # Run test policy script
- pixi run greedy         # Run greedy agent script

Or activate the environment interactively:
- pixi shell              # Enter the pixi environment
