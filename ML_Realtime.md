# Machine Learning Real-time

For the goal of this project, I needed to run and manipulate the trained neural nets in real-time. While I could have accomplished this with python and windowing libraries I decided to use [TouchDesigner](https://derivative.ca) (TD). What TD gained me was a graphical interface, faster iterations, and lots of built-in capabilities for audio, video, images, networking, etc. TD also has python as a scripting language making the integration of machine learning significantly easier. With regards to hardware, my only option was a Windows operating system with the best Nvidia graphics card I could afford.

## Installation

This

On the computer I was using, I first needed to remove the preinstalled Nvidia Frameview and disable the graphics driver and PhysX. The graphics driver that shipped with the computer was not compatible with the CUDA and PyTorch tools I needed to use. I downgraded my driver to v471.41. Not all computers will need to do this. I downloaded and installed the 11.2 version of the [CUDA Toolkit](https://developer.nvidia.com/cuda-11.2.0-download-archive) for windows. This [guide](https://docs.nvidia.com/cuda/archive/11.2.0/cuda-installation-guide-microsoft-windows/index.html) is also quite helpful

I installed a 3.9 version of [python](https://www.python.org/downloads/release/python-3913/). Again I was not intending to have multiple installs, so I did not use MiniConda or Conda. Also, for real-time use in TD, I found it easier to have the system path variables point at a single python install. Then installed the [pip](https://pip.pypa.io/en/stable/installation/) package manager. Next up, the PyTorch libs and the GPU CUDA handles. I used windows powershell to access the command line, because the commands are unix based rather than dos based. I ran the below command or could be acquired from [here](https://pytorch.org/get-started/locally/).
```
> pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Similarly to the linux install needed the gcc tools to compile some of the cuda files on demand. On windows, I installed [Visual Studio Community 2019](https://visualstudio.microsoft.com/vs/community/). Then, see image below, installed the _Desktop development with C++_ workload. After this, system variables need to be created. To do that, I ran the bat file at this location.
```
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat
```

Back to python again. The _trick_ to getting access and live, real-time neural net manipulation was using the CuPy library. I followed the instructions [here](https://docs.cupy.dev/en/stable/install.html) for the version of CUDA I have installed. The two additional libraries I used were [cuTENSOR](https://developer.nvidia.com/cutensor) and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html). Both may require creating an Nvidia developer account.


changes to stylegan3/utils/customops

