# CurbNet
A neural network designed to identify and segment curbs and cut curbs from street imagery. This project is a part of my bachelor's thesis at the Albert-Ludwigs-Universität Freiburg.

The goal is to create a network architecture that is well suited to efficiently and accurately identify curbs and [curb cuts](https://en.wikipedia.org/wiki/Curb_cut).
This network will then be used to aid in the pathfinding of the EUROPA2 (European Robotic Pedestrian Assitant 2.0) robot platform.

The project was motivated by the fact that currently our robotics platform is capable of identifying pedestrian safe zones and streets but if there is a blockage in the path, it would not be able to find a path that crosses the street.
This is to ensure that the robot does not fall over while traversing a curb.
The goal is to include this network as part of the pathfinding algorithm, allowing the robot to find a safe path down curb cuts and allowing it to cross streets.

![The EUROPA2 Robot Platform](https://github.com/yvan674/CurbNet/blob/master/media/europa2.jpg) 


## Installation
To install, first use conda to create a new environment either using the supplied curbnet.yml environment file or with the requirements file.
More complete instructions can be found in the [INSTALL.md](https://github.com/yvan674/CurbNet/blob/master/INSTALL.md) file.

## Usage
### Single Session
The proper way to use the network is by invoking the main_args.py script with:
```bash
main_args.py [arguments]
```
To view the possible arguments, simply invoke `main_args.py --h`.

Or by listing the configuration within a json file and feeding it into the main_json.py script as follows:
```bash
main_json.py [path to json file]
```
The proper format of the json file can be seen in the docstring for `main_json.py`.

Both main files have the added line:`#!/usr/bin/env python` so that they are directly executable.

If the command-line argument flag is unused or the property set to false, then a GUI will be invoked.
Note that command line operation is only available in nix-like operating systems, as it requires curses.

### Batch session
To run a batch session, run the command
```bash
batch_train.py [path to json file]
```
The json file should be formatted similarly to the json file used by main_json.py, but with one entry for each training session under the top-level key `"sessions"`.

### AutoML Optimization session
An AutoML optimization session can also be done using the BOHB (Bayesian Optimization and HyperBand) optimizer.
This is done by running
```bash
bohb/automl.py [dataset path] [output folder] [minimum budget] [maximum budget] [iterations]
```
Dataset path is the path to the dataset, the output folder is where the results (written as `results.pkl`) will be written to, the minimum/maximum budget is the minimum/maximum number of epochs to run each configuration for respectively, and the iterations is the number of configurations to try. 

## License
This project is released under the MIT License. Please review the [License](https://raw.githubusercontent.com/yvan674/CurbNet/master/LICENSE) file for more details.
