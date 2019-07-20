# Installation
This is a guide to installing this project as well as setting up the dataset.

We strongly recommend that you create a virtual environment using conda.

## Dependencies
### CUDA
This project was built to run on CUDA 9.0. It should work though on any other CUDA version that is compatible with Pytorch.
Install the CUDA toolkit as you would normally.

####Cuda without `sudo`
Occasionally, you might want to install the CUDA toolkit without using `sudo`, such as when you don't have `sudo` access to the computer.
This can get confusing. To help with that, here is a step-by-step guide on installing the CUDA toolkit without `sudo`.

1. Download the CUDA toolkit version that fits your version best from the [NVIDIA website](https://www.developer.nvidia.com/cuda-downloads).
You'll want to download the runfile.
Note that Pytorch only works with CUDA 9.0 or CUDA 10.0.
    
    1. Alternatively, you can also use `wget` to download the cuda toolkit.
    Use the command `wget <url to cuda runfile>`.
    You can get the runfile url by navigating to the website and right-clicking the download link and selecting copy linked address.

2. Once you've downloaded the runfile, make it executable by running `chmod +x cuda_<version>`

3. Now run the runfile with `./cuda_version`. 
In the interactive menu, when it asks `Enter Toolkit Location`, type a user writeable directory.
When it asks for a sample path, either use the default or enter another user writeable path.

4. Finally, a few lines must be added to your `.bashrc` file.
These lines are as follows, with `<toolkit path>` being where you installed the toolkit to:

```bash
export PATH=<toolkit path>/bin:${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<toolkit path>/lib64
```

You're done! Now confirm that the toolkit is installed by running the following command:
```bash
nvcc --version
```
This should return the toolkit version that you have just installed.

### Python dependencies
The required dependencies can be found in the file `requirements.txt`.

There might be issues with installing `imgaug`.
If this occurs, use the following command:
The requirement `imgaug` will most likely fail to install. To install this, while in the correct environment, use the following commands:
```bash
pip install git+https://github.com/aleju/imgaug
```

## Setting up the dataset
To set up the dataset, a few steps must be taken.
In general, these steps are:
1. Set up the directory structure
2. Select viable images
3. Optionally, resize the images

#### Setting up the directory structure
The directory must be set up such that it is as follows
```
main dataset directory/
├── training/
│   ├── images/
│   ├── instances/
│   ├── labels/
│   └── panoptic
├── validation/
│   ├── images/
│   ├── instances/
│   ├── labels/
│   └── panoptic
├── testing/
│   ├── images/
│   ├── instances/
│   ├── labels/
│   └── panoptic
```
The instances and panoptic directories of each subset is unnecessary, but included by default by the mapillary dataset.

#### Selecting viable images
Viable images (i.e. has at least a curb) are selected by using the script `utils/selector.py`.

It can be run on a dataset subfolder (i.e. training, validation, or testing) by using the following command
```bash
python utils/selector.py [absolute path to subfolder]
```
This will produce a file `viable.txt` in the subfolder, which contains a list of all viable images to use.

#### Optional: Resizing the images
Optionally, the images can also be resized.
If resized to fit within RAM, this reduces the number of file access operations required dramatically, increasing training speeds by at least 80%.
This is done running `utils/resize.py` similarly to the `utils/selector.py` script.
```bash
python utils/resize.py [absolute path to the subfolder]
``` 
This produces a folder called `[subfolder]_resized`, which contains the resized images while keeping the original directory structure (ignoring panoptic and instances).
The file `viable.txt` is still in the original subfolder and should not be deleted, although the original images may be deleted to conserve disk space.

