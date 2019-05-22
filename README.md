# CurbNetG
A neural network designed to identify and segment curbs and cut curbs from street imagery. This project is a part of my bachelor's thesis at the Albert-Ludwigs-Universität Freiburg.

The goal is to create a network architecture that is well suited to efficiently and accurately identify curbs and [curb cuts](https://en.wikipedia.org/wiki/Curb_cut). This network will then be used to aid in the pathfinding of the EUROPA2 (European Robotic Pedestrian Assitant 2.0) robot platform.

The project was motivated by the fact that currently our robotics platform is capable of identifying pedestrian safe zones and streets but if there is a blockage in the path, it would not be able to find a path that crosses the street. This is to ensure that the robot does not fall over while traversing a curb. The goal is to include this network as part of the pathfinding algorithm, allowing the robot to find a safe path down curb cuts and allowing it to cross streets.

![The EUROPA2 Robot Platform](https://github.com/yvan674/CurbNet/blob/master/media/europa2.jpg) 

## Usage
### Command line
The proper way to use the network is by invoking the main.py script with:
```bash
python3 main.py [arguments]
```

### GUI
Eventually, a GUI will also be implemented to allow for easier usage for training and inference.

## License
This project is released under the MIT License. Please review the [License](https://raw.githubusercontent.com/yvan674/CurbNet/master/LICENSE) file for more details.
