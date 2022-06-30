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
        me.mod.vars.G = pickle.load(f)['G_ema'].cuda()

    with open(relative_path, 'rb') as f:
        me.mod.vars.GHold = pickle.load(f)['G_ema'].cuda()

    relative_path = op('target').par.File2.eval()
    with open(relative_path, 'rb') as f:
        me.mod.vars.G2 = pickle.load(f)['G_ema'].cuda()

    me.mod.vars.G.eval()
    me.mod.vars.G2.eval()
    me.mod.vars.GHold.eval()
    me.mod.vars.gsd = me.mod.vars.G.state_dict()
    me.mod.vars.gsdHold = me.mod.vars.GHold.state_dict()
    me.mod.vars.gsd2 = me.mod.vars.G2.state_dict()
    return

def onCook(scriptOp):
    if me.mod.vars.G == None or me.mod.vars.G2 == None:
        return

    with torch.no_grad():
        src = source.cudaMemory()
        memPtr = cpCudaMem.MemoryPointer(
                    cpCudaMem.UnownedMemory(src.ptr, src.size, src),
                    0)

        r = cp.ndarray((1, 512), cp.float32, memPtr)
        r = torch.as_tensor(r, device='cuda')

        blend_layer_by_name('L12_1044_32', .5)
        me.mod.vars.G.load_state_dict(me.mod.vars.gsd)
        img = me.mod.vars.G(r, c)  #final image

        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0,2,3,1)
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

def blend_layer_by_name(name, value: float = 0.1):
    me.mod.vars.gsd['synthesis.'+name+'.weight'] = me.mod.vars.gsdHold['synthesis.'+name+'.weight'] * (1-value) + me.mod.vars.gsd2['synthesis.'+name+'.weight'] * value
    me.mod.vars.gsd['synthesis.'+name+'.bias'] = me.mod.vars.gsdHold['synthesis.'+name+'.bias'] * (1-value) + me.mod.vars.gsd2['synthesis.'+name+'.bias'] * value
    me.mod.vars.gsd['synthesis.'+name+'.magnitude_ema'] = me.mod.vars.gsdHold['synthesis.'+name+'.magnitude_ema'] * (1-value) + me.mod.vars.gsd2['synthesis.'+name+'.magnitude_ema'] * value
    me.mod.vars.gsd['synthesis.'+name+'.up_filter'] = me.mod.vars.gsdHold['synthesis.'+name+'.up_filter'] * (1-value) + me.mod.vars.gsd2['synthesis.'+name+'.up_filter'] * value
    me.mod.vars.gsd['synthesis.'+name+'.down_filter'] = me.mod.vars.gsdHold['synthesis.'+name+'.down_filter'] * (1-value) + me.mod.vars.gsd2['synthesis.'+name+'.down_filter'] * value
    me.mod.vars.gsd['synthesis.'+name+'.affine.weight'] = me.mod.vars.gsdHold['synthesis.'+name+'.affine.weight'] * (1-value) + me.mod.vars.gsd2['synthesis.'+name+'.affine.weight'] * value
    me.mod.vars.gsd['synthesis.'+name+'.affine.bias'] = me.mod.vars.gsdHold['synthesis.'+name+'.affine.bias'] * (1-value) + me.mod.vars.gsd2['synthesis.'+name+'.affine.bias'] * value
