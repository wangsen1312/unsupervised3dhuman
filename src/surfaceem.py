import os
import pickle
import torch
import smplx
import numpy as np
from src.customloss import body_fitting_loss_em
from src.prior import MaxMixturePrior

#  surface EM
class surface_EM_pt():
    """Implementation of SMPLify, use surface."""

    def __init__(self,
                 smplxmodel,
                 step_size=1e-1,
                 batch_size=1,
                 num_iters=100,
                 selected_index=np.arange(6890),
                 use_collision=False,
                 device=torch.device('cuda:0'),
                 GMM_MODEL_DIR="./smpl_models/",
                 mu = 0.02
                 ):

        # Store options
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=GMM_MODEL_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL-X model
        self.smpl = smplxmodel
        self.modelfaces = torch.from_numpy(np.int32(smplxmodel.faces)).to(device)
        self.selected_index = selected_index
        
        # mesh intersection
        self.model_faces = smplxmodel.faces_tensor.view(-1)
        self.use_collision = use_collision
        
        # outlier prob
        self.mu = mu
    
    @torch.no_grad()
    def prob_cal(self, modelVerts_in, meshVerts_in, sigma=0.05**2, mu = 0.02):
        modelVerts_sq = torch.squeeze(modelVerts_in) 
        meshVerts_sq = torch.squeeze(meshVerts_in)
        
        modelVerts = modelVerts_sq
        meshVerts = meshVerts_sq
        
        model_x, model_y, model_z = torch.split(modelVerts, [1,1,1], dim=1)
        mesh_x, mesh_y, mesh_z = torch.split(meshVerts, [1,1,1], dim=1)
        
        M = model_x.shape[0]
        N = mesh_x.shape[0]
            
        delta_x = torch.repeat_interleave(torch.transpose(mesh_x, 0, 1), M, dim=0) - torch.repeat_interleave(model_x, N, dim=1)
        delta_y = torch.repeat_interleave(torch.transpose(mesh_y, 0, 1), M, dim=0) - torch.repeat_interleave(model_y, N, dim=1)
        delta_z = torch.repeat_interleave(torch.transpose(mesh_z, 0, 1), M, dim=0) - torch.repeat_interleave(model_z, N, dim=1)
        
        deltaVerts= delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        
        sigmaInit = sigma #1e-3 # 1e-4
        d = 3.0 # three dimension
        mu_c = ((2.0 * torch.asin(torch.tensor(1.)) * sigmaInit)**(d/2.0) * mu * M)/((1-mu)*N)
        
        deltaExp  = torch.exp(-deltaVerts / (2*sigmaInit))
        deltaExpN = torch.repeat_interleave(torch.reshape(torch.sum(deltaExp, dim=0),(1, N)), M, dim=0)
        probArray = deltaExp / (deltaExpN + mu_c)
        
        Ind = torch.where(probArray > 1e-6) #2e-7
        modelInd, meshInd = Ind[0], Ind[1]
        probInput = probArray[Ind]
        
        #print(deltaVerts.shape)
        #print(probArray.shape)
        
        #P_sum  = torch.sum(probArray)
        #P_sep = torch.sum(probArray * deltaVerts)
        #sigma2 = P_sep/(P_sum*3)
                
        return probInput, modelInd, meshInd

    # ---- get the man function hrere
    def __call__(self, init_pose, init_betas, init_cam_t, meshVerts):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose
            init_betas: SMPL betas
            init_cam_t: Camera translation
            meshVerts: point3d from mesh
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """
        
        ### add the mesh inter-section to avoid
        search_tree = None
        pen_distance = None
        filter_faces = None

        # Make camera translation a learnable parameter
        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        
        camera_translation = init_cam_t.clone()
        
        betas = init_betas.detach().clone()
        preserve_betas = init_betas.detach().clone()
        preserve_pose = init_pose[:, 3:].detach().clone()

        # -------- Step : Optimize use surface points ---------
        betas.requires_grad = True
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True
        body_opt_params = [body_pose, global_orient, betas, camera_translation] # 

        # optimize the body_pose
        body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=20,
                                           lr=self.step_size, line_search_fn='strong_wolfe') #
        for i in range(self.num_iters):
            def closure():
                body_optimizer.zero_grad()
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas,
                                        transl=camera_translation,
                                        return_verts=True)
                
                modelVerts = smpl_output.vertices[:, self.selected_index]
                # calculate the probInput
                probInput, modelInd, meshInd = self.prob_cal(modelVerts, meshVerts, sigma=(0.15**2)*(self.num_iters-i+1)/self.num_iters, mu=self.mu) 
                #sigma=(0.1**2)*(self.num_iters-i+1)/self.num_iters

                loss =  body_fitting_loss_em(body_pose, preserve_pose, betas, preserve_betas, 
                                             camera_translation,
                                             modelVerts, meshVerts, modelInd, meshInd, probInput, 
                                             self.pose_prior,
                                             smpl_output, self.modelfaces, 
                                             pose_prior_weight=4.78*2.0, #avoid some collisions
                                             pose_preserve_weight=0.0,
                                             correspond_weight=1500.0,
                                             chamfer_weight=200.0,
                                             point2mesh_weight=0.0,
                                             use_collision=self.use_collision, 
                                             model_vertices=smpl_output.vertices, model_faces=self.model_faces,
                                             search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)
                                             
                loss.backward()
                return loss

            body_optimizer.step(closure)
            

            
        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas,
                                    transl=camera_translation,
                                    return_full_pose=True)
 
        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation
        
        
        
        
#  surface EM
class surface_EM_depth():
    """Implementation of SMPLify, use surface."""

    def __init__(self,
                 smplxmodel,
                 step_size=1e-1,
                 batch_size=1,
                 num_iters=100,
                 selected_index=np.arange(6890),
                 use_collision=False,
                 device=torch.device('cuda:0'),
                 GMM_MODEL_DIR="./smpl_models/",
                 mu=0.05
                 ):

        # Store options
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=GMM_MODEL_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL-X model
        self.smpl = smplxmodel
        self.modelfaces = torch.from_numpy(np.int32(smplxmodel.faces)).to(device)
        self.selected_index = selected_index
        
        # mesh intersection
        self.model_faces = smplxmodel.faces_tensor.view(-1)
        self.use_collision = use_collision
        
        # mu prob
        self.mu = mu
    
    @torch.no_grad()
    def prob_cal(self, modelVerts_in, meshVerts_in, sigma=0.05**2, mu = 0.02):
        modelVerts_sq = torch.squeeze(modelVerts_in) 
        meshVerts_sq = torch.squeeze(meshVerts_in)
        
        modelVerts = modelVerts_sq
        meshVerts = meshVerts_sq
        
        model_x, model_y, model_z = torch.split(modelVerts, [1,1,1], dim=1)
        mesh_x, mesh_y, mesh_z = torch.split(meshVerts, [1,1,1], dim=1)
        
        M = model_x.shape[0]
        N = mesh_x.shape[0]
            
        delta_x = torch.repeat_interleave(torch.transpose(mesh_x, 0, 1), M, dim=0) - torch.repeat_interleave(model_x, N, dim=1)
        delta_y = torch.repeat_interleave(torch.transpose(mesh_y, 0, 1), M, dim=0) - torch.repeat_interleave(model_y, N, dim=1)
        delta_z = torch.repeat_interleave(torch.transpose(mesh_z, 0, 1), M, dim=0) - torch.repeat_interleave(model_z, N, dim=1)
        
        deltaVerts= delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        
        sigmaInit = sigma
        d = 3.0 # three dimension
        mu_c = ((2.0 * torch.asin(torch.tensor(1.)) * sigmaInit)**(d/2.0) * mu * M)/((1-mu)*N)
        
        deltaExp  = torch.exp(-deltaVerts / (2*sigmaInit))
        deltaExpN = torch.repeat_interleave(torch.reshape(torch.sum(deltaExp, dim=0),(1, N)), M, dim=0)
        probArray = deltaExp / (deltaExpN + mu_c)
        
        Ind = torch.where(probArray > 1e-6) #2e-7
        modelInd, meshInd = Ind[0], Ind[1]
        probInput = probArray[Ind]

                
        return probInput, modelInd, meshInd

    # ---- get the man function hrere
    def __call__(self, init_pose, init_betas, init_cam_t, meshVerts):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose
            init_betas: SMPL betas
            init_cam_t: Camera translation
            meshVerts: point3d from mesh
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """
        
        ### add the mesh inter-section to avoid
        search_tree = None
        pen_distance = None
        filter_faces = None

        # Make camera translation a learnable parameter
        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        
        camera_translation = init_cam_t.clone()
        
        betas = init_betas.detach().clone()
        preserve_betas = init_betas.detach().clone()
        preserve_pose = init_pose[:, 3:].detach().clone()

        # -------- Step : Optimize use surface points ---------
        betas.requires_grad = True
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True
        body_opt_params = [body_pose, global_orient, betas, camera_translation] # 

        # optimize the body_pose
        body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=20,
                                           lr=self.step_size, line_search_fn='strong_wolfe') #
        for i in range(self.num_iters):
            def closure():
                body_optimizer.zero_grad()
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas,
                                        transl=camera_translation,
                                        return_verts=True)
                
                modelVerts = smpl_output.vertices[:, self.selected_index]
                # calculate the probInput
                probInput, modelInd, meshInd = self.prob_cal(modelVerts, meshVerts, sigma=(0.1**2)*(self.num_iters-i+1)/self.num_iters, mu=self.mu) 
                #sigma=(0.1**2)*(self.num_iters-i+1)/self.num_iters

                loss =  body_fitting_loss_em(body_pose, preserve_pose, betas, preserve_betas, 
                                             camera_translation,
                                             modelVerts, meshVerts, modelInd, meshInd, probInput, 
                                             self.pose_prior,
                                             smpl_output, self.modelfaces, 
                                             pose_prior_weight=4.78*3.0,
                                             pose_preserve_weight=5.0,
                                             correspond_weight=1000.0,
                                             chamfer_weight=100.0,
                                             point2mesh_weight=200.0,
                                             shape_prior_weight=2.0, 
                                             use_collision=self.use_collision, 
                                             model_vertices=smpl_output.vertices, model_faces=self.model_faces,
                                             search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)
                                             
                loss.backward()
                return loss

            body_optimizer.step(closure)
            

            
        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas,
                                    transl=camera_translation,
                                    return_full_pose=True)
                                                         
        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation
