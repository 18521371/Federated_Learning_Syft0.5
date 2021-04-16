import syft as sy

from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.core.plan.plan_builder import make_plan
from syft.lib.python.int import Int
from syft.lib.python.list import List
from syft.proto.core.plan.plan_pb2 import Plan as PlanPB
from syft.proto.lib.python.list_pb2 import List as ListPB

import torch as th

#from .training_helper import set_params, cross_entropy_loss, sgd_step
from . import local_model

# Set updated params to model
def set_params(model, params):
    for p, p_new in zip(model.parameters(), params):
        p.data = p_new.data

# Loss function
def cross_entropy_loss(logits, targets, batch_size):
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    return -(targets * log_probs).sum() / batch_size

# Optimizer - Update gradients
def sgd_step(model, lr=0.1):
    with ROOT_CLIENT.torch.no_grad():
        for p in model.parameters():
            p.data = p.data - lr * p.grad
            p.grad = th.zeros_like(p.grad.get())


# Training function define as plan - Build and send it into PyGrid (Domain)
# Clients perform training based on this plan
@make_plan
def training_plan(
    xs=th.rand(64, 1, 28, 28),
    ys=th.zeros([64, 10]),
    params=List(local_model.parameters())
):

    model = local_model.send(ROOT_CLIENT)
    set_params(model, params)
    out = model(xs)
    loss = cross_entropy_loss(out, ys, 64)
    loss.backward()
    sgd_step(model)

    return model.parameters(), loss


# Average function define as plan - Build and send it into PyGrid (Domain)
# Server perform aggregation based on this plan
@make_plan
def avg_plan(
    avg=List(local_model.parameters()), 
    item=List(local_model.parameters()), 
    num=Int(0)
):
    new_avg = []
    for i, param in enumerate(avg):
        new_avg.append((avg[i] * num + item[i]) / (num + 1))
    return new_avg

