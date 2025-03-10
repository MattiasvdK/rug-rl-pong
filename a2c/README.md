[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Ygf3q214)
# Final Assignment

In this assignment, you will implement various model-based algorithms. You can choose to either continue working with the Catch environment or swtich to a new one.

This template repository is almost the same as the previous one, with slight changes made to the Agent class.

If you decide to not use the Catch environment, make sure to adapt (or remove) the `environment.py` file.

## Installation

Before running the code, make sure you properly install the dependencies:

```
python3 -m pip install -r requirements.txt
```

The code has been tested with Python 3.11.

## Running the code

Now, you should be able to simply run the `main.py` as any other script.

In order to allow us for easy grading, we implemented a simple test that checks wheter your `requirements.txt` file is complete and that your code runs without errors. You can run this test by executing the following command:

```
source test.sh
```

## Tips and Resources

Here are a couple of hints and resources that might help you with in this assignment:

1. To help you out with technical writing, check out these papers for inspiration. Reading real scientific papers can help you out with using correct nomenclature and ensuring a clear structure. In particular, you can draw inspiration as to how complex concepts and formulas are introduced
   and explained.

   a. Technical Report on implementing RL algorithms in CartPole environment - https://arxiv.org/pdf/2006.04938.pdf

   b. Paper summarising usage of RL in Chess - https://page.mi.fu-berlin.de/block/concibe2008.pdf

2. If you have duplicate code in multiple places, it’s probably a bad sign. Maybe you should try it to group that functionality in a seperate function?
3. The agent should be able to learn using different types of algorithms. Maybe there is a way to make these algorithms easily swappable?
4. Type hinting is not required, but it can help your partner understand your code - https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
5. Git workshop by Cover - https://studysupport.svcover.nl/?a=1
6. YouTube Git tutorial - https://www.youtube.com/watch?v=RGOj5yH7evk
7. OOP in Python - https://www.youtube.com/watch?v=JeznW_7DlB0
8. How to document Python? - https://www.datacamp.com/tutorial/docstrings-python4

## Questions and help

If you are struggling with one part of the assignment, you're probably not alone. That's why we want to create a small FAQ throughout the next couple of weeks. In case of a question, raise an issue in the original, template repository: [https://github.com/Deep-Reinforcement-Learning-RUG/catch-assignment](https://github.com/Deep-Reinforcement-Learning-RUG/catch-assignment). We will answer your questions there, so that there are no duplicate questions.

## Notes of the submitter

Most of my code is directly reused from the first assignment.
The main changes are the (obvious) agent classes and the simulator.
The simulator now runs the Catch environment one game at a time, for a specific
amount of games instead of timesteps. I also added the GIFWriter that writes
episodes to gif files in the `./gifs/` folder. 

From here on the README from the first assignment with some changes to fit this
one.

## Running the Code

The code can be run through calling the main function. To run different agents, the name of the
agent to the arguments:

```
python3 ./main.py <agent>
```

e.g.:

```
python3 ./main.py a2c
```

If no agent name is provided the `a2c` agent is used. Providing any other name
than `a2c` is also futile as it is the only implemented agent.

The agent names are:
* `a2c`: Advantage Actor-Critic (`a2cagent.py`)


**It is important to note that running an agent overwrites the results written for that agent.
The program will ask for confirmation before overwriting the results.**

## The Main Loop

The main scripts runs the simulator together with the agent factory.
The agent factory implements a grid-search over the hyperparameters defined
in `hyperparameters.py`.

## Important Added Classes

AgentFactory (`agentfactory.py`): Generates instances of the provided Agent class based on grid-search over the provided parameters in `hyperparameters.py`.

Simulator (`simulator.py`): Runs the agents generated by the AgentFactory for a given number of simulations and episodes.
It calculates the results of terminal states and logs those through the CSVWriter class

CSVWriter (`csvwriter.py`): Writes the logged results and hyperparameters to csv files in `./data/`
Hyperparameters are logged in `./data/<agent>_params.csv` while the results for each different instance
is logged in the files `./data/<agent>_results_<identifier>.csv` where the identifier is a unique index
of the agent with the given hyperparameters. The results of an agent defined in the hyperparameter file
can thus be linked through the identifier.

GIFWriter (`gifwriter.py`): Writes episodes to gif files. This class was mainly
implemented to create visualizations for the presentation.

## Some Notes

* It might have been nice if the writers shared a base class for the Writer classes, but due to time limitations
and the nature of the GIFWriter class I opted for taking a bit of a shortcut.
* Since I only implemented one agent (A2C) the separation between `Agent` and `A2CAgent` became
a bit blurred. As far as I can tell the separation of responsibility is quite well, however
it might occur that some functionality would have been better suited in the other class
if more agents were to be implemented.
