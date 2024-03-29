# Machine Learning - Training

This Guide explains how to train [Stylegan 3](https://github.com/NVlabs/stylegan3).

## Setup

I used [Linode](https://www.linode.com), because I already have a web server with them and I'm familiar with using the terminal. I am faculty in the [School of Art](https://art.asu.edu) at Arizona State University (ASU). I have access to ASU's [Computing Services](https://cores.research.asu.edu/research-computing/about), and I was able to train when resources were available. I found Amazon AWS was more cumbersome than needed. Google Colab would timeout and run based on available resources, so this wasn't reliable for the long periods that I needed.

In Linode, I created an Ubuntu 22.04 LTS instance with 32GB of RAM, 640GB SSD HD, and Nvidia RTX6000 GPU. This sounds impressive, but is the smallest and least expensive option. Once this was up and running I used the terminal to ssh into the computer.

## Install

Once connected to this remote computer, I needed to install all of the NVidia CUDA requirements. Thankfully Linode had a [guide](https://www.linode.com/docs/products/compute/gpu/guides/install-nvidia-cuda/) for this.

Because I'm connected to this remote computer by a terminal, when it came to installing the CUDA Toolkit I chose the deb (network) option and ran those commands provided.

After this, I needed to install Python and required necessary libraries. An alternative method is with MiniConda, but in this situation I was only going to have one Python install and then delete this machine instance as soon as I was done.

```bash
> sudo apt install software-properties-common
> sudo add-apt-repository ppa:deadsnakes/ppa
> sudo apt install python3.9
> sudo apt install python3-pip
```

Now onto [PyTorch](https://pytorch.org/get-started/locally/)

```bash
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

Git was already installed, so we only need to clone Stylegan 3 into the place we want.

```bash
> git clone https://github.com/NVlabs/stylegan3.git
```

Because I did not use MiniConda or Conda, I need to install dependencies on my own. Looking at the _environment.yml_ file from the stylegan 3 repository I ran this:

```bash
> pip3 install numpy click pillow scipy requests tqdm ninja matplotlib imageio
```

Then ran pip3 install %whatever I'm missing% to fix things as errors came up.


## Test

Lastly, a test to make sure this all works. From inside the stylegan 3 repository we can run their gen_images script.
```bash
> python3 gen_images.py --outdir=out --trunc=1 --seeds=2 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```

Install anything that might be needed. An output image should be created in the output folder when successful.

## Training

I created a couple of extra folders on the server to manage my data and checkpoints. Then, I used [Filezilla](https://filezilla-project.org) in sftp mode to transfer my datasets to my server. From inside the stylegan 3 repository I ran their training script with _nohup_ in front so that if my connection closed or the local computer went to sleep, the remote training would continue.

```bash 
> nohup python3 train.py \
--outdir=/root/checkpoints \
--data=/root/datasets/set05.zip \
--cfg=stylegan3-t \
--gpus=1 \
--batch=32 \
--batch-gpu=4 \
--snap=5 \
--gamma=32 \
--kimg=200 \
--mirror=0
```

These batch settings are for a V100. Once this is running, we can follow the output by running this:

```bash
> tail -f nohup.out
```

And we can always end the process by checking with the command _top_ to see the process id and then _kill -pid_.

## Wait

Make coffee.
