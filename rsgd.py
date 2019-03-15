#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch as th
from torch.optim.optimizer import Optimizer, required
import model
from model import HyperboloidDistance
import numpy as np
import global_variables

spten_t = th.sparse.FloatTensor

def project_tensors_onto_tangent_space(hyperboloid_points, ambient_gradients):
    """
        project gradients in the ambiant space onto the tangent space
    :param hyperboloid_point: A point on the hyperboloid
    :param ambient_gradient: The gradient to project
    :return: gradients in the tangent spaces of the hyperboloid points
        """
    lhs=model.HyperboloidDistance.minkowski_tensor_dot(hyperboloid_points, ambient_gradients)

    rhs=hyperboloid_points
    lhs=lhs.type(rhs.type())
    scaled_points=th.mul(lhs,rhs)

    nonzero_points=th.nonzero(th.sum(scaled_points,1))
    ind=nonzero_points[0]
    tangent_gradients=ambient_gradients + scaled_points
    return tangent_gradients

def transform_grads(grad):
    """
        multiply by the inverse of the Minkowski metric tensor g = diag[-1, 1,1 ... 1] to make the first element of each
        grad vector negative
    :param grad: grad matrix of shape (n_vars, embedding_dim)
    :return:
        """
    # FIXME change hardcoded shape!
    x = np.eye(global_variables.N_DIM)
    x[0, 0] = -1.
    # T = th.Tensor(x, dtype=grad.dtype)
    # T = th.FloatTensor(x, dtype=grad.dtype)
    # T = th.from_numpy(np.array(x,dtype=np.float32))

    T = th.from_numpy(np.array(x)).cuda(grad.device.index)
    T=T.type(grad.type())
    return th.matmul(grad, T)

def hyperboloid_grad(hyperboloid_point, ambient_gradient):
    r"""
    Function to compute Riemannian gradient from the
    Euclidean gradient in the Poincaré ball.

    Args:
        hyperboloid_point (Tensor): Current point in the ball
        ambient_gradient (Tensor): Ambient gradient at p
    """

    minkowski_grads = transform_grads(ambient_gradient)
    # print (hyperboloid_point.shape)
    # print (ambient_gradient.shape)

    # tangent_grads = project_tensors_onto_tangent_space(hyperboloid_point, ambient_gradient)
    tangent_grads = project_tensors_onto_tangent_space(hyperboloid_point, minkowski_grads)

    # sys.exit(0)
    return tangent_grads

def tensor_exp_map(hyperboloid_points, tangent_grads):
    """
        Map vectors in the tangent space of the hyperboloid points back onto the hyperboloid
    :param hyperboloid_points: a tensor of points on the hyperboloid of shape (#examples, #dims)
    :param tangent_grads: a tensor of gradients on the tangent spaces of the hyperboloid_points of shape (#examples, #dims)
    :return:
        """
    # norms=th.norm(tangent_grads,p=2,dim=1, keepdim=True)
    # print ("nb out of l2 ball TANGENT GRADIENT [%i] out of [%i]" % (th.sum(norms>global_variables.L2_GRAD_CLIP), th.sum(tangent_grads>0)))
    # # out_of_l2ball_inds=th.squeeze(norms>1)
    # out_of_l2ball_inds=th.squeeze(norms>global_variables.L2_GRAD_CLIP)
    # print (th.max(norms))
    # tangent_grads[out_of_l2ball_inds]=th.div(tangent_grads[out_of_l2ball_inds], norms[out_of_l2ball_inds])
    # norms=th.norm(tangent_grads,p=2,dim=1, keepdim=True)
    # print (th.max(norms))

    norms_squared=model.HyperboloidDistance.minkowski_tensor_dot(tangent_grads, tangent_grads).float()
    # print ("Opt step - exp map - tangent gradient squared norm")
    # print (norms_squared.shape)
    # print (norms_squared.detach().cpu().numpy())
    # print (th.sum(th.abs(norms_squared)))

    zeros= th.zeros(norms_squared.size()).cuda(hyperboloid_points.device.index)
    nonzero_flags = th.squeeze(th.ne(norms_squared, zeros))
    nonzero_indices = th.squeeze(th.nonzero(nonzero_flags))
    # nonzero_norms_squared=norms_squared[nonzero_flags]
    nonzero_norms=th.sqrt(norms_squared[nonzero_flags])

    # print ("Opt step - exp map - tangent gradient non zero norm")
    # print (nonzero_norms.shape)
    # print (nonzero_norms.detach().cpu().numpy())
    # print (th.sum(th.abs(nonzero_norms)))

    updated_grads = tangent_grads[th.squeeze(nonzero_flags)]

    updated_points = hyperboloid_points[nonzero_flags]
    nonzero_norms=nonzero_norms.type(updated_grads.type())
    updated_points=updated_points.type(nonzero_norms.type())

    lhs=th.mul(th.cosh(nonzero_norms), updated_points)

    normed_grads = th.div(updated_grads, nonzero_norms+global_variables.MY_EPS)
    # normed_grads = th.div(updated_grads, th.clamp(nonzero_norms, min=1e-5))
    normed_grads=normed_grads.type(nonzero_norms.type())

    rhs=th.mul(th.sinh(nonzero_norms), normed_grads)
    updates = rhs+lhs
    saved_coords=hyperboloid_points.data[nonzero_indices].clone().detach()
    indsnan_mask=th.isnan(hyperboloid_points.data[nonzero_indices]) | th.isinf(hyperboloid_points.data[nonzero_indices])
    # indsnan_mask=th.isnan(updates) | th.isinf(updates)
    # print ("nb nan in exp map updates [%i] out of [%i]" % (th.sum(indsnan_mask), th.numel(indsnan_mask)))
    # print ("before")
    # print (indsnan_mask)
    # hyperboloid_points[nonzero_indices].data=updates
    hyperboloid_points.data[nonzero_indices]=th.where(indsnan_mask, saved_coords, updates)
    # hyperboloid_points.data[nonzero_indices]=th.where(indsnan_mask, updates, saved_coords)
    # print ("after")
    # indsnan_mask=th.isnan(hyperboloid_points.data[nonzero_indices]) | th.isinf(hyperboloid_points.data[nonzero_indices])
    # print (indsnan_mask)
    # hyperboloid_points.data[nonzero_indices][:,0]=th.sqrt(th.sum(th.pow(hyperboloid_points.data[nonzero_indices][:,1:],2), 1)+1.)

    # print ("nb out of box values in exp map updates [%i] out of [%i]" % (th.sum(th.abs(hyperboloid_points.data[nonzero_indices])>100), th.numel(hyperboloid_points.data[nonzero_indices])))
    # hyperboloid_points.data[nonzero_indices]=th.clamp(hyperboloid_points.data[nonzero_indices], min=-100,max=100)

    # w=hyperboloid_points.data[nonzero_indices]
    # norms=th.norm(w,p=2,dim=1, keepdim=True)
    # print ("nb out of l2 ball in exp map updates [%i] out of [%i]" % (th.sum(norms>1), th.numel(w)))
    # out_of_l2ball_inds=th.squeeze(norms>1)
    # print (th.max(norms))
    # w[out_of_l2ball_inds]=th.div(w[out_of_l2ball_inds],norms[out_of_l2ball_inds])
    # norms=th.norm(w,p=2,dim=1, keepdim=True)
    # print (th.max(norms))
    # hyperboloid_points.data[nonzero_indices]=w
    # norms=th.norm(hyperboloid_points.data[nonzero_indices],p=2,dim=1, keepdim=True)
    # print (th.max(norms))

    if len(hyperboloid_points.data[nonzero_indices]):
        hyperboloid_points.data[nonzero_indices][:,0]=th.sqrt(th.sum(th.pow(hyperboloid_points.data[nonzero_indices][:,1:],2), 1)+1.)
    # print ("final")
    # indsnan_mask=th.isnan(hyperboloid_points.data[nonzero_indices]) | th.isinf(hyperboloid_points.data[nonzero_indices])
    if th.any(th.isnan(hyperboloid_points.data[nonzero_indices]) | th.isinf(hyperboloid_points.data[nonzero_indices])):
        print ("nan or inf after exp map")
        sys.exit(33)
    # print (indsnan_mask)

def hyperboloid_retraction(hyperboloid_point, tangent_gradient, lr):
    tensor_exp_map(hyperboloid_point, -lr * tangent_gradient)

def poincare_grad(p, d_p):
    r"""
    Function to compute Riemannian gradient from the
    Euclidean gradient in the Poincaré ball.

    Args:
        p (Tensor): Current point in the ball
        d_p (Tensor): Euclidean gradient at p
    """
    if d_p.is_sparse:
        p_sqnorm = th.sum(
            p.data[d_p._indices()[0].squeeze()] ** 2, dim=1,
            keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p

def euclidean_grad(p, d_p):
    return d_p

def euclidean_retraction(p, d_p, lr):
    p.data.add_(-lr, d_p)

class RiemannianSGD(Optimizer):
    r"""Riemannian stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rgrad (Function): Function to compute the Riemannian gradient from
            an Euclidean gradient
        retraction (Function): Function to update the parameters via a
            retraction of the Riemannian gradient
        lr (float): learning rate
    """

    def __init__(self, params, lr=required, rgrad=required, retraction=required):
        defaults = dict(lr=lr, rgrad=rgrad, retraction=retraction)
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """Performs a single optimization step.

        Arguments:
s            lr (float, optional): learning rate for the current update.
        """
        loss = None

        i = 0
        # print ("Opt step - parameters")
        # print (len(self.param_groups))
        # print (self.param_groups[0].keys())
        # print (type(self.param_groups[0]["params"][0]))
        # print (self.param_groups[0]["params"][0])
        # sys.exit(1)
        for group in self.param_groups:
            # print ("group[%i]" %i )
            # print (group)
            # print ("--")
            j = 0
            for p in group['params'][0:]:
                # print ("group[%i] - param %i"  % (i,j))
                # print (p.shape)
                # print (p)
                # print ("***")
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # print ("Opt step - gradients in ambient")
                # np.set_printoptions(suppress=False)
                # print (d_p.shape)
                # print (d_p.cpu().numpy())
                # print (th.sum(th.abs(d_p)))
                # print ("----")
                # print(group)
                # print (p)
                # sys.exit(0)
                if lr is None:
                    lr = group['lr']
                d_p = group['rgrad'](p, d_p)
                group['retraction'](p, d_p, lr)
                # sys.exit(3)
                # if j == 0:
                #     p.data=th.clamp(p.data, max=-1e-8)
                    # p.data=th.clamp(p.data, min=-1e-8)
                    # p=th.clamp(p, min=-1e-8)
                    # print (p)
                j = j + 1
            i = i + 1

        return loss
