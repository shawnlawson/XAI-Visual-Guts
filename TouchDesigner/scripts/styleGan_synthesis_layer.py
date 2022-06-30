# me - this DAT
# scriptOp - the OP which is cooking
import sys
sys.path.insert(0, "C:/Users/Shawn Lawson/AppData/Local/Programs/Python/Python39/Lib/site-packages")
sys.path.insert(1, "C:/Users/Shawn Lawson/Documents/Github/stylegan3")
sys.path.append("C:/Users/Shawn Lawson/AppData/Local/Programs/Python/Python39/include")
sys.path.append("C:/Users/Shawn Lawson/AppData/Local/Programs/Python/libs")

import torch
import torch_utils
import pickle
import dnnlib
import legacy
import numpy as np
import cupy as cp

source = op('source')
target = op('target')

cpCudaMem = cp.cuda.memory
c = None

def onSetupParameters(scriptOp):
    return

def onPulse(par):
    relative_path = op('target').par.File.eval()
    with open(relative_path, 'rb') as f:
        me.mod.vars.G = legacy.load_network_pkl(f)['G_ema'].cuda() # for SG2 ADA

    me.mod.vars.G.eval()

    return

def onCook(scriptOp):
    if me.mod.vars.G == None:
        return

    with torch.no_grad():
        src = source.cudaMemory()
        memPtr = cpCudaMem.MemoryPointer(
                    cpCudaMem.UnownedMemory(src.ptr, src.size, src),
                    0)

        r = cp.ndarray((1, 512), cp.float32, memPtr)
        r = torch.as_tensor(r, device='cuda')

        img = me.mod.vars.G.synthesis.input(r)
        for name in me.mod.vars.G.synthesis.layer_names:
            if 'L6' in name: # L0-L14, input
                break
            img = getattr(me.mod.vars.G.synthesis, name)(img, r)
        
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        dim = img.size(dim=1)

        img = img[:,:,:,:3]

        img = cp.asarray(img[0])
        a = cp.ndarray((dim, dim, 1), dtype=cp.uint8)
        a.fill(255)
        img = cp.dstack((img, a))

        size = dim*dim*4
        src.shape.width = dim
        src.shape.height = dim
        src.shape.dataType = np.uint8
        src.shape.numComps = 4
        target.copyCUDAMemory(img.data.ptr, size, src.shape)

    return
