import emd
import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Function





class emdFunction(Function):
    @staticmethod
    def forward(ctx, da1, dda2, eps, iters):

        batchsize, n, _ = da1.size()
        _, m, _ = dda2.size()

        assert(n == m)
        assert(da1.size()[0] == dda2.size()[0])
        assert(n % 1024 == 0)
        assert(batchsize <= 512)

        da1 = da1.contiguous().float().cuda()
        dda2 = dda2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(da1, dda2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(da1, dda2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        da1, dda2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradda1 = torch.zeros(da1.size(), device='cuda').contiguous()
        graddda2 = torch.zeros(dda2.size(), device='cuda').contiguous()

        emd.backward(da1, dda2, gradda1, graddist, assignment)
        return gradda1, graddda2, None, None

class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)

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
        
