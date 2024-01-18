# SimHuman
 
## Installation
**Clone this repo:**
```bash
git clone https://github.com/neuralchen/SimBeauty_Dev.git
cd SimBeauty_Dev
```



## Dependencies

### Python dependencies
All dependencies for defining the environment are provided in ```environment.yaml```. We recommend running this repository using Anaconda (you may need to modify environment.yml to install PyTorch that matches your own CUDA version following https://pytorch.org/):

```bash
conda env create -f environment.yml
```
### TensorRT
TensorRT download URL: ```https://developer.nvidia.com/tensorrt-download```

***CUDA version must match with TensorRT version.***

Untar TensorRT installation package:

```bash
tar xzvf TensorRT-xxxxxxxxxxxx.tar
```

Add system environment PATH in ```/etc/profile``` or ```~/.bashrc```

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[path to your tensorrt]/lib
```

Copy files to system lib:

```bash
sudo cp -r [your tensorrt dir]/lib/* /usr/lib
sudo cp -r [your tensorrt dir]/include/* /usr/include
```

Installation of TensorRT, ***Before installation, please check the python version***:

```bash
# TensorRT installation
cd [your tensorrt dir]/python
pip install tensorrt-xxxxxxxx-py2.py3-none-any.whl

# UFF installation to support model conversion of tensorflow
cd [your tensorrt dir]/uff
pip install uff-xxxxxxxx-py2.py3-none-any.whl

# graphsurgeon installation, to support self-defined operators and structures
cd [your tensorrt dir]/graphsurgeon
pip install graphsurgeon-xxxxxxxxxx-py2.py3-none-any.whl
```

To verify installation, please import tensorrt and uff in python env.

### Checkpoints:

#### Face Detection

#### TTS

#### Face Restoration

#### Lip Models


