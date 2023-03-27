import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.cuda.amp import autocast

from emd import emd

class EmdFunction(Function):
    """
    PyTorch autograd Function wrapper for the emd CUDA extension
    """
    @staticmethod
    def forward(ctx, da1, dda2, eps, iters):
        with torch.cuda.amp.autocast():
            batchsize, n, _ = da1.size()
            _, m, _ = dda2.size()

            assert(n == m)
            assert(da1.size()[0] == dda2.size()[0])
            assert(n % 1024 == 0)
            assert(batchsize <= 512)

            device = da1.device
            dist = torch.zeros(batchsize, n, device=device)
            assignment = -torch.ones(batchsize, n, device=device, dtype=torch.int32)
            assignment_inv = -torch.ones(batchsize, m, device=device, dtype=torch.int32)
            price = torch.zeros(batchsize, m, device=device)
            bid = torch.zeros(batchsize, n, device=device, dtype=torch.int32)
            bid_increments = torch.zeros(batchsize, n, device=device)
            max_increments = torch.zeros(batchsize, m, device=device)
            unass_idx = torch.zeros(batchsize * n, device=device, dtype=torch.int32)
            max_idx = torch.zeros(batchsize * m, device=device, dtype=torch.int32)
            unass_cnt = torch.zeros(512, dtype=torch.int32, device=device)
            unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device=device)
            cnt_tmp = torch.zeros(512, dtype=torch.int32, device=device)

            emd.forward(da1.float(), dda2.float(), dist, assignment, price, assignment_inv, bid,
                        bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum,
                        cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(da1, dda2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        with torch.cuda.amp.autocast():
            da1, dda2, assignment = ctx.saved_tensors
            gradda1 = torch.zeros(da1.size(), device=da1.device)
            graddda2 = torch.zeros(dda2.size(), device=dda2.device)

            emd.backward(da1.float(), dda2.float(), gradda1, graddist, assignment)
        return gradda1, graddda2, None, None

def test_emd():
    x1 = torch.rand(20, 8192, 3).cuda() # please normalize your point cloud to [0, 1]
    x2 = torch.rand(20, 8192, 3).cuda()
    emd = emdModule()
    start_time = time.perf_counter()
    dis, assigment = emd(x1, x2, 0.002, 10000) # 0.005, 50 for training 
    print("Input_size: ", x1.shape)
    print("Runtime: %lfs" % (time.perf_counter() - start_time))
    print("EMD: %lf" % np.sqrt(dis.cpu()).mean())
    print("|set(assignment)|: %d" % assigment.unique().numel())
    assigment = assigment.cpu().numpy()
    assigment = np.expand_dims(assigment, -1)
    x2 = np.take_along_axis(x2, assigment, axis = 1)
    d = (x1 - x2) * (x1 - x2)
    print("Verified EMD: %lf" % np.sqrt(d.cpu().sum(-1)).mean())

#test_emd()
        
