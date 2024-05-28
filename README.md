# SimHuman
 
## Installation
**Clone this repo:**
```bash
git clone https://github.com/neuralchen/SimHuman.git
cd SimHuman
```

## Dependencies

### Python dependencies
All dependencies for defining the environment are provided in ```environment.yml```. We recommend running this repository using Anaconda (you may need to modify environment.yml to install PyTorch that matches your own CUDA version following https://pytorch.org/):

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
Download face detection models from [Google Driver](https://drive.google.com/file/d/1amwJw2Oiq2OIocHjjKBnByLy7dqkCFAN/view?usp=sharing).

- glintr100.onnx        --->  ./insightface_func/models/antelope/
- scrfd_10g_bnkps.onnx  --->  ./insightface_func/models/antelope/
- det_10g.onnx          --->  ./insightface_func/new/

#### TTS
Download TTS models from [[Baidu Driver](https://pan.baidu.com/s/1aizIt1Hg9Xhb1ttCrbzOvQ), Password：```qgpi```]

- 8000.pth.tar                  --->  ./TTS/output/ckpt/biaobei/
- generator_universal.pth.tar   --->  ./TTS/hifigan/
- best_model.pt                 --->  ./TTS/transformer/prosody_model/

#### Face Restoration
[GFPGANv1.4.onnx(324M)](https://drive.google.com/file/d/1yeR0-YuzoEulzZP1NZkhN3KNMfFfH8Rb/view?usp=sharing)

- GFPGANv1.4.onnx                  --->  ./restoration/

#### Lip Models
[wav2lip.pth(416M)](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)

- wav2lip.pth                 --->  ./checkpoints/


## Usage

```python
python gui.py --video ./example_videos/example.mp4
```
