import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory

from models.dataset import Dataset, Dataset_nv
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, DeformNetwork, NGPNetwork, AppearanceNetwork, TopoNetwork, DeformNetwork_MLP
from models.renderer import NeuSRenderer, DeformNeRFRenderer
from models.dataset_hyper import iPhoneDatasetFromAllFrames
from torch.utils.data import DataLoader
from models.NeRF import NerfMLP,NerfMLPpp
from models.schedule import * 
class Runner:
    def __init__(self,dataset,  conf_path, mode='train', case='CASE_NAME', is_continue=False, is_paint = False, iswoab = False):
        self.device = torch.device('cuda')
        self.gpu = torch.cuda.current_device()
        self.dtype = torch.get_default_dtype()
        self.is_paint = is_paint
        # Configuration
        self.mode = mode
        self.hyper = (dataset!='ndr')
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        if is_continue == False and self.is_paint==True:
            self.base_exp_dir += 'wocontinue'
        if self.conf.get_bool('model.deform_network.normalizing_flow') == False: 
            self.base_exp_dir = self.base_exp_dir + '_deform_mlp'
        if self.conf.get_bool('dataset.use_depth') == False: 
            self.base_exp_dir = self.base_exp_dir + '_wodepth'
        if iswoab==True: 
            self.base_exp_dir = self.base_exp_dir + '_woab'
        self.base_exp_dir = self.base_exp_dir + '_NeRF'
        self.pre_image = 0
        os.makedirs(self.base_exp_dir, exist_ok=True)
        if self.hyper == False:
            self.dataset = Dataset(self.conf['dataset'])
        else:
            if dataset != 'dycheck':
                self.dataset = iPhoneDatasetFromAllFrames(self.conf['dataset'], split = 'train_intl', use_depth = False)
            else:
                self.dataset = iPhoneDatasetFromAllFrames(self.conf['dataset'], split = 'train')

      
            
        assert iswoab != self.conf.get_bool('model.topo_network.isuse')
        if is_paint ==True:
            self.conf['dataset']['data_dir'] = self.conf['dataset']['data_dir'].replace('_paint', '')
            if self.hyper == False:
                self.dataset_val = Dataset(self.conf['dataset'])
            else:
                if dataset != 'dycheck':
                    self.dataset_val = iPhoneDatasetFromAllFrames(self.conf['dataset'], split = 'val_intl', use_depth = False)
                else:
                    self.dataset_val = iPhoneDatasetFromAllFrames(self.conf['dataset'], split = 'train')
        else:
            if self.hyper == False:
                self.dataset_val = Dataset(self.conf['dataset'])
            else:
                if dataset != 'dycheck':
                    self.dataset_val = iPhoneDatasetFromAllFrames(self.conf['dataset'], split = 'val_intl', use_depth = False)
                else:
                    self.dataset_val = iPhoneDatasetFromAllFrames(self.conf['dataset'], split = 'train')
                
        if self.hyper == False:
            self.dataset_nv = Dataset_nv(self.conf['dataset'])
        else:
            if dataset != 'dycheck':
                self.dataset_nv = iPhoneDatasetFromAllFrames(self.conf['dataset'], split = 'val_intl', use_depth = False)
            else:
                self.dataset_nv = iPhoneDatasetFromAllFrames(self.conf['dataset'], split = 'train')

        self.iter_step = 0

        # Deform
        self.use_deform = self.conf.get_bool('train.use_deform')
        if self.use_deform:
            self.deform_dim = self.conf.get_int('model.deform_network.d_feature')
            self.deform_codes = torch.nn.Embedding(self.dataset.n_images, self.deform_dim) #torch.randn(self.dataset.n_images, self.deform_dim, requires_grad=True).to(self.device)
            self.deform_codes.weight = torch.nn.init.uniform_(self.deform_codes.weight, b = 0.05)
            self.appearance_dim = self.conf.get_int('model.appearance_rendering_network.d_global_feature')
            self.appearance_codes = torch.nn.Embedding(self.dataset.n_images, self.appearance_dim) #torch.randn(self.dataset.n_images, self.appearance_dim, requires_grad=True).to(self.device)

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        if is_paint == True:
            self.end_iter += 40000
        self.important_begin_iter = self.conf.get_int('model.neus_renderer.important_begin_iter')
        # Anneal
        self.max_pe_iter = self.conf.get_int('train.max_pe_iter')

        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')

        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.validate_idx = 0
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_final_value = self.conf.get_float('train.learning_rate_final_value')
        self.lr_num_steps = self.conf.get_float('train.lr_num_steps')

        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.test_batch_size = self.conf.get_int('test.test_batch_size')

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Depth
        self.use_depth = self.conf.get_bool('dataset.use_depth')
        if self.use_depth:
            self.geo_weight = self.conf.get_float('train.geo_weight')
            self.angle_weight = self.conf.get_float('train.angle_weight')

        # Deform
        if self.use_deform:
            if self.conf.get_bool('model.deform_network.normalizing_flow') == True:
                self.deform_network = DeformNetwork(**self.conf['model.deform_network']).to(self.device)
            else:
                self.deform_network = DeformNetwork_MLP(**self.conf['model.deform_network']).to(self.device)

            self.topo_network = TopoNetwork(**self.conf['model.topo_network']).to(self.device)
            
        self.sdf_network = NerfMLPpp(**self.conf['model.sdf_network']).to(self.device)

        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        # Deform
        # if self.use_deform:
        #     if self.is_paint==False:
        #         self.color_network = AppearanceNetwork(**self.conf['model.appearance_rendering_network']).to(self.device)
        #     else:
        #         self.ngp_color = NGPNetwork().to(self.device)
        #         self.color_network = AppearanceNetwork(**self.conf['model.appearance_rendering_network']).to(self.device)

        # else:
        #     self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        # Deform
        if self.use_deform:
            if self.is_paint == True:
                self.renderer = DeformNeRFRenderer(self.report_freq,
                                        self.deform_network,
                                        self.topo_network,
                                        self.sdf_network,
                                        self.deviation_network,
                                        # self.color_network,
                                        ngp_color = self.ngp_color,
                                        **self.conf['model.neus_renderer'])
            else:
                self.renderer = DeformNeRFRenderer(self.report_freq,
                                        self.deform_network,
                                        self.topo_network,
                                        self.sdf_network,
                                        self.deviation_network,
                                        # self.color_network,
                                        **self.conf['model.neus_renderer'])
        else:
            self.renderer = NeuSRenderer(self.sdf_network,
                                        self.deviation_network,
                                        # self.color_network,
                                        **self.conf['model.neus_renderer'])

        # Load Optimizer
        params_to_train = []
        if is_paint==False:
            if self.use_deform:
                params_to_train += [{'name':'deform_network', 'params':self.deform_network.parameters(), 'lr':self.learning_rate}]
                params_to_train += [{'name':'topo_network', 'params':self.topo_network.parameters(), 'lr':self.learning_rate}]
                params_to_train += [{'name':'deform_codes', 'params':self.deform_codes.parameters(), 'lr':self.learning_rate}]
                params_to_train += [{'name':'appearance_codes', 'params':self.appearance_codes.parameters(), 'lr':self.learning_rate}]
            params_to_train += [{'name':'sdf_network', 'params':self.sdf_network.parameters(), 'lr':self.learning_rate}]
            params_to_train += [{'name':'deviation_network', 'params':self.deviation_network.parameters(), 'lr':self.learning_rate}]
            # params_to_train += [{'name':'color_network', 'params':self.color_network.parameters(), 'lr':self.learning_rate}]
            # params_to_train += [{'name':'ngp_color', 'params':self.ngp_color.parameters(), 'lr':self.learning_rate * 100}]
            if self.dataset.camera_trainable:
                params_to_train += [{'name':'intrinsics_paras', 'params':self.dataset.intrinsics_paras, 'lr':self.learning_rate}]
                params_to_train += [{'name':'poses_paras', 'params':self.dataset.poses_paras, 'lr':self.learning_rate}]
                # Depth
                if self.use_depth:
                    params_to_train += [{'name':'depth_intrinsics_paras', 'params':self.dataset.depth_intrinsics_paras, 'lr':self.learning_rate}]
        else:
            for q in self.deform_network.parameters():
                q.requires_grad = False
            for q in self.topo_network.parameters():
                q.requires_grad = False
            self.deform_codes.requires_grad = False
            self.appearance_codes.requires_grad = False
            for q in self.sdf_network.parameters():
                q.requires_grad = False
            for q in self.deviation_network.parameters():
                q.requires_grad = False
            # for q in self.color_network.parameters():
            #     q.requires_grad = False
            params_to_train += [{'name':'color_network', 'params':self.color_network.parameters(), 'lr':self.learning_rate}]
            # params_to_train += [{'name':'sdf_network', 'params':self.sdf_network.parameters(), 'lr':self.learning_rate}]

        # Camera
        self.optimizer = torch.optim.Adam(params_to_train)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            # if self.mode == 'validate_pretrained':
            #     latest_model_name = 'pretrained.pth'
            # else:
            if is_paint == True and mode!='render':
                if iswoab == False:
                    ckpts_path = self.base_exp_dir.replace("_paint", "")
                else:
                    ckpts_path = self.base_exp_dir.replace("_paint", "_woab")
            elif mode == 'render':
                ckpts_path = self.base_exp_dir.replace("result", "result_video1")
            else:
                ckpts_path =  self.base_exp_dir
            model_list_raw = os.listdir(os.path.join(ckpts_path, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name, ckpts_path)
            
        self.base_exp_dir = self.base_exp_dir + '_mlp++'
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # Backup codes and configs
        if self.mode[:5] == 'train':
            self.file_backup()
        # if is_paint == True:
        #     self.base_exp_dir = self.base_exp_dir + '_paint'
        # os.makedirs(self.base_exp_dir, exist_ok=True)
        
        
        res_step = self.end_iter - self.iter_step

        self.dataset.len = res_step
        self.dataset.batch_size = self.batch_size
        self.train_loader = DataLoader(dataset=self.dataset,
                          batch_size=1,
                          num_workers=4,
                          generator=torch.Generator(device='cuda'),
                          shuffle=True)
        self.expschedule = ExponentialSchedule(self.learning_rate, self.learning_rate_final_value, self.lr_num_steps)
        self.warp_alpha_sch = LinearSchedule(self.conf.get_float('train.warp_alpha_initial_value'), self.conf.get_float('train.warp_alpha_final_value'), self.conf.get_float('train.warp_alpha_num_steps'))
        self.ambient_alpha_sch = LinearSchedule(self.conf.get_float('train.ambient_alpha_initial_value'), self.conf.get_float('train.ambient_alpha_final_value'), self.conf.get_float('train.ambient_alpha_num_steps'), self.conf.get_float('train.ambient_alpha_start_iter'))

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        if self.is_paint == False:
            self.update_learning_rate()
            self.update_alpha()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        # self.validate_all_image(resolution_level=1)
        with tqdm(total=len(self.train_loader)) as t:
            for iter_i, data in enumerate(self.train_loader):
                # Deform
                
                image_idx = image_perm[self.iter_step % len(image_perm)]
                # Deform
                if iter_i == 0:
                    print('The files will be saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                self.iter_step += 1

                    # self.validate_observation_mesh(self.validate_idx)
                # Depth
                # if self.use_depth:
                    # if self.is_paint == False:
                    #     data = self.dataset.gen_random_rays_at_depth(image_idx, self.batch_size)
                    # else:
                    #     data = self.dataset.gen_random_rays_at_depth_paint(image_idx, self.batch_size)
                    # rays_o, rays_d, rays_s, rays_l, true_rgb, mask = \
                    #     data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 13], data[:, 13: 14]
                # else:
                #     data = self.dataset.gen_random_rays_at(image_idx, self.batch_size)
                data = data.to(self.device)[0]
                rays_o, rays_d, true_rgb, mask, time_id = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, -1:]

                # Deform
                deform_code = self.deform_codes#[time_id.long()][None, ...]

                appearance_code = self.appearance_codes#[time_id.long()][None, ...]
                # Anneal
                
                alpha_ratio = max(min(self.iter_step/self.max_pe_iter, 1.), 0.)
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d, self.hyper)
                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).to(self.dtype)
                else:
                    mask = torch.ones_like(mask)
                mask_sum = mask.sum() + 1e-5
                
                render_out = self.renderer.render(time_id.long(), deform_code, appearance_code, rays_o, rays_d, near, far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                alpha_ratio=alpha_ratio, iter_step=self.iter_step)
                # Depth
                
                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                gradient_o_error = render_out['gradient_o_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                depth_map = render_out['depth_map']

                # Loss
                color_error = (color_fine - true_rgb)  ** 2 * mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum')/ (mask_sum * 3)
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())
                # mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-5, 1.0 - 1e-5), mask)
                # Depth
                loss = color_fine_loss #+\
                        # ( mask_loss * self.mask_weight) * regular_scale

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.iter_step % 500 == 0:
                    self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                    self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                    del color_fine_loss
                    self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                    self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                    self.writer.add_scalar('Statistics/deform_codes',  list(self.deform_codes.parameters())[0].mean(), self.iter_step)
                    self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
                    self.writer.add_scalar('Statistics/warp_alpha',  self.deform_network.warp_em.alpha.weights.item(), self.iter_step)
                    self.writer.add_scalar('Statistics/amb_alpha', self.sdf_network.amb.alpha.weights.item(), self.iter_step)
                    self.writer.add_scalar('Statistics/lr',  self.expschedule.get(self.iter_step), self.iter_step)
                # self.writer.add_scalar('Loss/parameters', list(self.color_network.parameters())[0].mean(), self.iter_step)
                if self.iter_step % self.report_freq == 0:
                    print('The files have been saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    print('iter:{:8>d} loss={} idx={} alpha_ratio={} lr={}'.format(self.iter_step, loss, image_idx,
                            alpha_ratio, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()
                if (self.iter_step % self.val_freq == 0) or ((self.iter_step-1) % 1000 == 0 and self.is_paint==True):
                    self.validate_image(self.validate_idx, resolution_level=1)
                    self.validate_idx += 1
                    if self.is_paint == True:
                        self.validate_image(0, resolution_level=1, val=False)

                    # Depth
                    if self.use_depth:
                        self.validate_image_with_depth(self.validate_idx, resolution_level=1)
                        self.validate_idx += 1

                        if self.is_paint == True:
                            self.validate_image_with_depth(0, resolution_level=1, val=False)


                # if self.iter_step % self.val_mesh_freq == 0:
                #     self.validate_observation_mesh(self.validate_idx)

                self.update_learning_rate()
                self.update_alpha()
                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()
                t.set_postfix(steps=self.iter_step, psnr=psnr.item())
                t.update(1)
                    
        self.save_checkpoint()
        self.validate_all_image(resolution_level=1)

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)


    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])


    def update_learning_rate(self, scale_factor=1):
        current_learning_rate = self.expschedule.get(self.iter_step)
        for g in self.optimizer.param_groups:
            g['lr'] = current_learning_rate

    def update_alpha(self, scale_factor=1):
        warp_alpha = self.warp_alpha_sch.get(self.iter_step)
        self.deform_network.warp_em.alpha.weights = torch.tensor(warp_alpha).to(self.device)
        amb_alpha = self.ambient_alpha_sch.get(self.iter_step)
        self.sdf_network.amb.alpha.weights = torch.tensor(amb_alpha).to(self.device)
        
    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))
        logging.info('File Saved')
        logging.info('save path:{}'.format(self.base_exp_dir))


    def load_checkpoint(self, checkpoint_name, ckpt_path):
        checkpoint = torch.load(os.path.join(ckpt_path, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'],strict=False)
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        # self.color_network.load_state_dict(checkpoint['color_network_fine'])
        if self.mode == 'render':
            self.ngp_color.load_state_dict(checkpoint['ngp_color'])
        # Deform
        if self.use_deform:

            self.deform_network.load_state_dict(checkpoint['deform_network'])
            self.topo_network.load_state_dict(checkpoint['topo_network'])
            self.deform_codes.load_state_dict(checkpoint['deform_codes'])
            self.appearance_codes.load_state_dict(checkpoint['appearance_codes'])
            logging.info('Use_deform True')
        if self.hyper==False:
            self.dataset.intrinsics_paras = torch.from_numpy(checkpoint['intrinsics_paras']).to(self.device)
            self.dataset.poses_paras = torch.from_numpy(checkpoint['poses_paras']).to(self.device)
        # Depth
            if self.use_depth:
                self.dataset.depth_intrinsics_paras = torch.from_numpy(checkpoint['depth_intrinsics_paras']).to(self.device)
            # Camera
        
            if self.dataset.camera_trainable:
                self.dataset.intrinsics_paras.requires_grad_()
                self.dataset.poses_paras.requires_grad_()
                # Depth
                if self.use_depth:
                    self.dataset.depth_intrinsics_paras.requires_grad_()
            else:
                self.dataset.static_paras_to_mat()
        # if self.is_paint == False:
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')


    def save_checkpoint(self):
        # Depth
        if self.use_depth and self.hyper==False:
            depth_intrinsics_paras = self.dataset.depth_intrinsics_paras.data.cpu().numpy()
        elif self.hyper==False:
            depth_intrinsics_paras = self.dataset.intrinsics_paras.data.cpu().numpy()
        else:
            depth_intrinsics_paras = None
        # Deform
        if self.use_deform:
            checkpoint = {
                'deform_network': self.deform_network.state_dict(),
                'topo_network': self.topo_network.state_dict(),
                'sdf_network_fine': self.sdf_network.state_dict(),
                'variance_network_fine': self.deviation_network.state_dict(),
                # 'color_network_fine': self.color_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
                'deform_codes': self.deform_codes.state_dict(),
                'appearance_codes': self.appearance_codes.state_dict(),
                'depth_intrinsics_paras': depth_intrinsics_paras,
            }
            if self.is_paint == True:
                checkpoint.update({'ngp_color': self.ngp_color.state_dict()})
            if self.hyper == False:
                checkpoint.update({'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy(),
                                    'poses_paras': self.dataset.poses_paras.data.cpu().numpy()})
        else:
            checkpoint = {
                # 'ngp_color': self.ngp_color.state_dict(),
                'sdf_network_fine': self.sdf_network.state_dict(),
                'variance_network_fine': self.deviation_network.state_dict(),
                # 'color_network_fine': self.color_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
                'depth_intrinsics_paras': depth_intrinsics_paras,
            }
            if self.is_paint == True:
                checkpoint.update({'ngp_color': self.ngp_color.state_dict(),
                                    'poses_paras': self.dataset.poses_paras.data.cpu().numpy()})
            if self.hyper == False:
                checkpoint.update({'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy()})
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>7d}.pth'.format(self.iter_step)))


    def validate_image(self, idx=-1, resolution_level=-1, mode='train', normal_filename='normals', rgb_filename='rgbs', depth_filename='depths', val = True, nv=False):
        if val==True:
            dataset_val = self.dataset_val
        else:
            dataset_val = self.dataset
        if nv ==True:
            dataset_val = self.dataset_nv

        if idx < 0:
            idx = np.random.randint(dataset_val.n_images)
        # Deform
 

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if mode == 'train':
            batch_size = self.batch_size
        else:
            batch_size = self.test_batch_size

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, time_id = dataset_val.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        time_ids = torch.ones((rays_o.reshape(-1, 3).shape[0],)).to(rays_o).long() * time_id
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)
        time_ids = time_ids.split(batch_size)
        
        out_rgb_fine = []
        out_normal_fine = []
        out_depth_fine = []

        for rays_o_batch, rays_d_batch, time_id in zip(rays_o, rays_d, time_ids):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch, self.hyper)

            if self.use_deform:
                render_out = self.renderer.render(time_id, self.deform_codes,
                                                self.appearance_codes,
                                                rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                alpha_ratio=max(min(self.iter_step/self.max_pe_iter, 1.), 0.),
                                                iter_step=self.iter_step)
                render_out['gradients'] = render_out['gradients_o']
            else:
                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio())
            
            def feasible(key): return (key in render_out) and (render_out[key] is not None)
            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                
            if feasible('gradients') and feasible('weights'):
                if self.iter_step >= self.important_begin_iter:
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                else:
                    n_samples = self.renderer.n_samples
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]

                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible('depth_map'):
                out_depth_fine.append(render_out['depth_map'].detach().cpu().numpy())
            del render_out
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
        if idx == 0:
            logging.info('diff:{}'.format((self.pre_image-img_fine).mean()))
            self.pre_image = img_fine

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # Camera
            if self.hyper == False:
                if dataset_val.camera_trainable:
                    _, pose = dataset_val.dynamic_paras_to_mat(idx)
                else:
                    pose = dataset_val.poses_all[idx]
                rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
                normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                            .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
            else:
                normal_img = (normal_img[:, :, None].reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
        depth_img = None
        if len(out_depth_fine) > 0:
            depth_img = np.concatenate(out_depth_fine, axis=0)
            depth_img = depth_img.reshape([H, W, 1, -1])
            depth_img = 255. - np.clip(depth_img/depth_img.max(), a_max=1, a_min=0) * 255.
            depth_img = np.uint8(depth_img)
        os.makedirs(os.path.join(self.base_exp_dir, rgb_filename), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, normal_filename), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, depth_filename), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        rgb_filename,
                                        '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                           np.concatenate([img_fine[..., i],
                                           dataset_val.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        normal_filename,
                                        '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                           normal_img[..., i])
            
            if len(out_depth_fine) > 0:
                if self.use_depth:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            depth_filename,
                                            '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                            np.concatenate([cv.applyColorMap(depth_img[..., i], cv.COLORMAP_JET),
                                                dataset_val.depth_at(idx, resolution_level=resolution_level)]))
                else:
                    cv.imwrite(os.path.join(self.base_exp_dir, depth_filename,
                                            '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                                            cv.applyColorMap(depth_img[..., i], cv.COLORMAP_JET))
        logging.info('save: {}'.format(self.base_exp_dir))

    def validate_image_with_depth(self, idx=-1, resolution_level=-1, mode='train', val=True):
        if val==True:
            dataset_val = self.dataset_val
        else:
            dataset_val = self.dataset

        if idx < 0:
            idx = np.random.randint(dataset_val.n_images)

        # Deform
        if self.use_deform:
            deform_code = self.deform_codes[idx][None, ...]
            appearance_code = self.appearance_codes[idx][None, ...]
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if mode == 'train':
            batch_size = self.batch_size
        else:
            batch_size = self.test_batch_size

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, rays_s, mask = dataset_val.gen_rays_at_depth(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)
        rays_s = rays_s.reshape(-1, 3).split(batch_size)
        mask = (mask > 0.5).to(self.dtype).detach().cpu().numpy()[..., None]

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch, rays_s_batch in zip(rays_o, rays_d, rays_s):
            color_batch, gradients_batch = self.renderer.renderondepth(deform_code,
                                                    appearance_code,
                                                    rays_o_batch,
                                                    rays_d_batch,
                                                    rays_s_batch,
                                                    alpha_ratio=max(min(self.iter_step/self.max_pe_iter, 1.), 0.))

            out_rgb_fine.append(color_batch.detach().cpu().numpy())
            out_normal_fine.append(gradients_batch.detach().cpu().numpy())
            del color_batch, gradients_batch

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            img_fine = img_fine * mask

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # w/ pose -> w/o pose. similar: world -> camera
            # Camera
            if self.hyper == False:
                if self.dataset.camera_trainable:
                    _, pose = self.dataset.dynamic_paras_to_mat(idx)
                else:
                    pose = self.dataset.poses_all[idx]
                rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
                normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                            .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
                normal_img = normal_img * mask
            else:
                normal_img = (normal_img[:, :, None].reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
                normal_img = normal_img * mask

        os.makedirs(os.path.join(self.base_exp_dir, 'rgbsondepth'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normalsondepth'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'rgbsondepth',
                                        '{:0>8d}_depth_{}.png'.format(self.iter_step, idx)),
                           np.concatenate([img_fine[..., i],
                                           dataset_val.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normalsondepth',
                                        '{:0>8d}_depth_{}.png'.format(self.iter_step, idx)),
                           normal_img[..., i])


    def validate_all_image(self, resolution_level=-1):
        for image_idx in range(self.dataset_val.n_images):
            self.validate_image(image_idx, resolution_level, 'test', 'validations_normals', 'validations_rgbs', 'validations_depths', val=True)
            print('Used GPU:', self.gpu)


    def validate_all_image_novel_view(self, resolution_level=-1):
        for image_idx in range(self.dataset_val.n_images):
            self.validate_image(image_idx, resolution_level, 'test', 'validations_normals_novel_veiws', 'validations_rgbs_novel_veiws', 'validations_depths_novel_veiws', val=True, nv=True)
            print('Used GPU:', self.gpu)
            
    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


    # Deform
    def validate_canonical_mesh(self, world_space=False, resolution=64, threshold=0.0, filename='meshes_canonical'):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        
        vertices, triangles =\
            self.renderer.extract_canonical_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold,
                                                        alpha_ratio=max(min(self.iter_step/self.max_pe_iter, 1.), 0.))
        os.makedirs(os.path.join(self.base_exp_dir, filename), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, filename, '{:0>8d}_canonical.ply'.format(self.iter_step)))

        logging.info('End')

    
    # Deform
    def validate_observation_mesh(self, idx=-1, world_space=False, resolution=64, threshold=0.0, filename='meshes'):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        # Deform
        deform_code = self.deform_codes[idx][None, ...]
        bound_min = torch.tensor(self.dataset.parser._bbox[0], dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.parser._bbox[1], dtype=self.dtype)
        
        vertices, triangles =\
            self.renderer.extract_observation_geometry(deform_code, bound_min, bound_max, resolution=resolution, threshold=threshold,
                                                        alpha_ratio=max(min(self.iter_step/self.max_pe_iter, 1.), 0.))
        os.makedirs(os.path.join(self.base_exp_dir, filename), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, filename, '{:0>8d}_{}.ply'.format(self.iter_step, idx)))

        logging.info('End')


    # Deform
    def validate_all_mesh(self, world_space=False, resolution=64, threshold=0.0):
        for image_idx in range(self.dataset.n_images):
            self.validate_observation_mesh(image_idx, world_space, resolution, threshold, 'validations_meshes')
            print('Used GPU:', self.gpu)



# This implementation is built upon NeuS: https://github.com/Totoro97/NeuS
if __name__ == '__main__':
    print('Welcome to NDR')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--paint', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--woab', default=False, action="store_true")
    parser.add_argument('--dataset', type=str, default="ndr")

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.cuda.set_device(args.gpu)

    runner = Runner(args.dataset, args.conf, args.mode, args.case, args.is_continue, args.paint, args.woab)

    if args.mode == 'train':
        runner.train()
        if args.paint == True:
            runner.validate_all_image(resolution_level=1)
    elif args.mode == 'render':
        runner.validate_all_image_novel_view(resolution_level=1)
        runner.validate_all_image(resolution_level=1)

    elif args.mode[:8] == 'validate':
        if runner.use_deform:
            runner.validate_all_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
            runner.validate_all_image(resolution_level=1)
        else:
            runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
            runner.validate_all_image(resolution_level=1)
