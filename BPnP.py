'''
MIT License

Copyright (c) 2019 Bo Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import cv2 as cv
import numpy as np
import utils.train_dsac_util as T



class BPnP(torch.autograd.Function):
    """
    Back-propagatable PnP
    INPUTS:
    pts2d - the 2D keypoints coordinates of size [batch_size, num_keypoints, 2]
    pts3d - the 3D keypoints coordinates of size [num_keypoints, 3]
    K     - the camera intrinsic matrix of size [3, 3]
    OUTPUT:
    P_6d  - the 6 DOF poses of size [batch_size, 6], where the first 3 elements of each row are the angle-axis rotation 
    vector (Euler vector) and the last 3 elements are the translation vector. 
    NOTE:
    This BPnP function assumes that all sets of 2D points in the mini-batch correspond to one common set of 3D points. 
    For situations where pts3d is also a mini-batch, use the BPnP_m3d class.
    """



    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        bs = pts2d.size(0)
        n = pts2d.size(1)
        device = pts2d.device
        pts3d_np = np.array(pts3d.detach().cpu())
        K_np = np.array(K.detach().cpu())
        P_6d = torch.zeros(bs,6,device=device)

        for i in range(bs):
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n,1,2))
            if ini_pose is None:
                success, rvec0, T0, _ = cv.solvePnPRansac(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv.SOLVEPNP_P3P, confidence=0.9999, reprojectionError=10, iterationsCount=10000)
                if not success:
                    raise ValueError('PnP gave us nones during ransac')
            else:
                rvec0 = np.array(ini_pose[i, 0:3].detach().cpu().reshape(3, 1))
                T0 = np.array(ini_pose[i, 3:6].detach().cpu().reshape(3, 1))

            success, rvec, T = cv.solvePnP(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE,
                                           useExtrinsicGuess=True, rvec=rvec0, tvec=T0)
            if not success:
                raise ValueError('PnP gave us nones during refinement')


            angle_axis = torch.tensor(rvec,device=device,dtype=torch.float).reshape(1, 3)
            T = torch.tensor(T,device=device,dtype=torch.float).reshape(1, 3)
            P_6d[i,:] = torch.cat((angle_axis,T),dim=-1)

        ctx.save_for_backward(pts2d,P_6d,pts3d,K)
        return P_6d

    @staticmethod
    def backward(ctx, grad_output):

        if (grad_output.nonzero().nelement() == 0):
            return None, None, None, None

        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)
        m = 6

        #grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        #grad_K = torch.zeros_like(K)

        for i in range(bs):
            J_fy = torch.zeros(m,m, device=device)
            #J_fx = torch.zeros(m,2*n, device=device)
            J_fz = torch.zeros(m,3*n, device=device)
            #J_fK = torch.zeros(m, 9, device=device)

            coefs = get_coefs(P_6d[i].reshape(1,6), pts3d, K)

            pts2d_flat = pts2d[i].clone().reshape(-1).detach().requires_grad_()
            P_6d_flat = P_6d[i].clone().reshape(-1).detach().requires_grad_()
            pts3d_flat = pts3d.clone().reshape(-1).detach().requires_grad_()
            K_flat = K.clone().reshape(-1).detach().requires_grad_()

            for j in range(m):
                torch.set_grad_enabled(True)
                if j > 0:
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                #R = kn.angle_axis_to_rotation_matrix(P_6d_flat[0:m-3].reshape(1,3))
                #R = rot_vector_to_matrix(P_6d_flat[0:m - 3].reshape(1, 3))
                R = T.Rodrigues(P_6d_flat[0:m - 3].reshape(1, 3).t(), 1, device)

                #P = torch.cat((R[0,0:3,0:3].reshape(3,3), P_6d_flat[m-3:m].reshape(3,1)),dim=-1)
                P = torch.cat((R[0:3, 0:3].reshape(3, 3), P_6d_flat[m - 3:m].reshape(3, 1)), dim=-1)
                KP = torch.mm(K_flat.reshape(3,3), P)
                pts2d_i = pts2d_flat.reshape(n,2).transpose(0,1)
                pts3d_i = torch.cat((pts3d_flat.reshape(n,3),torch.ones(n,1,device=device)),dim=-1).t()
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2,:].reshape(1,n)

                r = pts2d_i*Si-proj_i[0:2,:]
                coef = coefs[:,:,j].transpose(0,1) # size: [2,n]
                fj = (coef*r).sum()
                fj.backward()
                J_fy[j,:] = P_6d_flat.grad.clone()
                #J_fx[j,:] = pts2d_flat.grad.clone()
                J_fz[j,:] = pts3d_flat.grad.clone()
                #J_fK[j,:] = K_flat.grad.clone()

            inv_J_fy = torch.inverse(J_fy)

            #J_yx = (-1) * torch.mm(inv_J_fy, J_fx)
            J_yz = (-1) * torch.mm(inv_J_fy, J_fz)
            #J_yK = (-1) * torch.mm(inv_J_fy, J_fK)

            #grad_x[i] = grad_output[i].reshape(1,m).mm(J_yx).reshape(n,2)
            grad_z += grad_output[i].reshape(1,m).mm(J_yz).reshape(n,3)
            #grad_K += grad_output[i].reshape(1,m).mm(J_yK).reshape(3,3)

        #grad_x is the gradient wrt the image coords
        #grad_z is the gradient wrt the scene coords
        #grad_K is the gradient wrt the camera matrix

        return None, grad_z, None, None


def get_coefs(P_6d, pts3d, K):
    device = P_6d.device
    n = pts3d.size(0)
    m = P_6d.size(-1)
    coefs = torch.zeros(n,2,m,device=device)
    torch.set_grad_enabled(True)
    y = P_6d.clone().repeat(n,1).detach().requires_grad_()
    proj = batch_project(y, pts3d.detach(), K.detach()).squeeze()
    vec = torch.diag(torch.ones(n,device=device).float())
    for k in range(2):
        torch.set_grad_enabled(True)
        y_grad = torch.autograd.grad(proj[:,:,k],y,vec, retain_graph=True)
        coefs[:,k,:] = -2*y_grad[0].clone()
    return coefs

def batch_project(P, pts3d, K, angle_axis=True):
    n = pts3d.size(0)
    bs = P.size(0)
    device = P.device
    pts3d_h = torch.cat((pts3d, torch.ones(n, 1, device=device)), dim=-1)
    if angle_axis:
        #R_out = kn.angle_axis_to_rotation_matrix(P[:, 0:3].reshape(bs, 3))
        R_out = T.Rodrigues(P[:, 0:3].reshape(bs, 3).t(), bs, device).reshape(bs, 3, 3)
        PM = torch.cat((R_out[:,0:3,0:3], P[:, 3:6].reshape(bs, 3, 1)), dim=-1)
    else:
        PM = P
    pts3d_cam = pts3d_h.matmul(PM.transpose(1,2))
    pts2d_proj = pts3d_cam.matmul(K.t())
    S = pts2d_proj[:,:, 2].reshape(bs, n, 1)
    pts2d_pro = pts2d_proj[:,:,0:2].div(S)

    return pts2d_pro





