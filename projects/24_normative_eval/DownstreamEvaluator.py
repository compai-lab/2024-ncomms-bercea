import logging
from torch.nn import L1Loss
import copy
import torch
import numpy as np
import cv2
import glob


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns
from PIL import Image
#
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score, roc_curve
import optim.losses.FID.fid_score as fid
#
import lpips
from model_zoo.vgg import VGGEncoder
import os
#
from optim.metrics import *
from core.DownstreamEvaluator import DownstreamEvaluator
#
from transforms.synthetic import *


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, task='thresholding',
                 normal_path='./weights/24_normative_eval/Normal_brain_train/*.jpeg'):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.compute_scores = True
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.global_ = False
        self.task = task
        self.training_normal_images_path = normal_path
        self.checkpoint_path = checkpoint_path + '/images/'

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        # DIFFERENT THRESHOLDS THAT CAN BE DERIVED USING THE THRESHOLDING FUNCTION ON HEALTHY IMAGES
        # th = 0.007 # RA
        # th = 0.081 # DAE
        # th = 0.241 # AnoDDPM Gaussian
        # th = 0.006 # AnoDDPM Gaussian PL
        # th = 0.188 # AnoDDPM Simplex
        # th = 0.004 # AnoDDPM Simplex PL
        # th = 0.198 # VQVAE
        # th = 0.214 # CE-VAE
        # th = 0.229 # LTM 0.212
        # th = 0.246 # VAE
        # th = 0.113 # AE-S
        # th = 0.307 # SI-VAE
        # th = 0.171 # MorphAEus
        th = 0.001 # AutoDDPM
        # th = 0.27 # MAE
        # th = 0.083  # Patched_anoDDPM
        # th = 0.296 # f-AnoGAN
        # th = 0.281
        # th = 0.234 # AnoDDPM Gaussian +
        # self.print_files(global_model)
        # if self.task == 'thresholding':
        #     _ = self.thresholding(global_model)
        # elif self.task == 'RQI':
        #     self.normative_eval_i_RQI(global_model)
        # elif self.task == 'AHI_UNN':
        #     self.normative_eval_ii_AHI_UN_N(global_model)
        # elif self.task == 'AHI':
        #     self.normative_eval_ii_AHI(global_model)
        # elif self.task == 'CACI':
        #     self.normative_eval_iii_CACI(global_model)
        # elif self.task == 'detection':
        self.object_localization(global_model, th)

    def _log_visualization(self, to_visualize, dataset_key, count):
        """
        Helper function to log images and masks to wandb
        :param: to_visualize: list of dicts of images and their configs to be visualized
            dict needs to include:
            - tensor: image tensor
            dict may include:
            - title: title of image
            - cmap: matplotlib colormap name
            - vmin: minimum value for colorbar
            - vmax: maximum value for colorbar
        :param: epoch: current epoch
        """
        diffp, axarr = plt.subplots(1, len(to_visualize), gridspec_kw={'wspace': 0, 'hspace': 0},
                                    figsize=(len(to_visualize) * 4, 4))
        for idx, dict in enumerate(to_visualize):
            if 'title' in dict:
                axarr[idx].set_title(dict['title'])
            axarr[idx].axis('off')
            tensor = dict['tensor'].cpu().detach().numpy().squeeze() if isinstance(dict['tensor'], torch.Tensor) else \
            dict['tensor'].squeeze()
            axarr[idx].imshow(tensor, cmap=dict.get('cmap', 'gray'), vmin=dict.get('vmin', 0), vmax=dict.get('vmax', 1))
        diffp.set_size_inches(len(to_visualize) * 4, 4)

        wandb.log({f'Anomaly_masks/Example_FastMRI_{dataset_key}_{count}': [wandb.Image(diffp, caption="Fast_" + str(
            count))]})

    def normative_eval_i_RQI(self, global_model):
        """
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        logging.info("################ Normative Evaluation: Req i. RQI #################")
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': []
        }

        #  INFERENCE FOR THE DATASETS
        for dataset_key in self.test_data_dict.keys():
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': []
            }
            dataset = self.test_data_dict[dataset_key]
            logging.info(f"#################################### {dataset_key} ############################################")

            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):

                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                x = data0.to(self.device)

                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)

                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), torch.zeros(x_i.shape)).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()

                    test_metrics['MAE'].append(np.mean(np.abs(x_rec_ - x_)))
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.npy', x_rec_)

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

            logging.info('Writing plots...')
            for metric in metrics:
                fig_bp = go.Figure()
                x = []
                y = []
                for idx, dataset_values in enumerate(metrics[metric]):
                    dataset_name = list(self.test_data_dict)[idx]
                    for dataset_val in dataset_values:
                        y.append(dataset_val)
                        x.append(dataset_name)

                fig_bp.add_trace(go.Box(
                    y=y,
                    x=x,
                    name=metric,
                    boxmean='sd'
                ))
                title = 'score'
                fig_bp.update_layout(
                    yaxis_title=title,
                    boxmode='group',  # group together boxes of the different traces for each value of x
                    yaxis=dict(range=[0, 1]),
                )
                fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

                wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})

    def normative_eval_ii_AHI_UN_N(self, global_model):
        """
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        dims = 2048
        num_workers = 4
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        logging.info("################ Normative Evaluation: Req ii. AHI UN; N #################")
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()

        #  TAKE THE TRAIN HEALTHY IMAGES AS REFERENCE
        train_healthy_path = self.training_normal_images_path
        train_healthy = []
        for img_path in glob.glob(train_healthy_path):
            im = Image.open(img_path)
            train_healthy.append(np.squeeze(np.asarray(im)) / 255.0)
        len_train_healthy = len(train_healthy)
        train_healthy = np.asarray(train_healthy)

        #  INFERENCE FOR THE DATASETS
        for dataset_key in self.test_data_dict.keys():
            input_images = []
            dataset = self.test_data_dict[dataset_key]
            logging.info(f"#################################### {dataset_key} ############################################")

            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                x = data0.to(self.device)

                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0].cpu().detach().numpy()
                    input_images.append(x_i)

            len_test_ = len(input_images)
            batch_size = len_test_

            nr_repeat = int(len_train_healthy / len_test_) + 1
            repeated_input = copy.deepcopy(input_images)
            repeated_input = np.repeat(np.asarray(repeated_input), nr_repeat, axis=0)
            repeated_input = repeated_input[:len_train_healthy, :, :]

            fid_UNN = fid.calculate_fid_given_images([repeated_input, train_healthy],
                                                    batch_size, device, dims, num_workers)


            logging.info('[ {} ]: FID_UN;N: {}'.format(dataset_key, fid_UNN))

    def normative_eval_ii_AHI(self, global_model):
        """
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        dims = 2048
        num_workers = 4
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        logging.info("################ Normative Evaluation: Req ii. AHI #################")
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()

        #  TAKE THE TRAIN HEALTHY IMAGES AS REFERENCE
        train_healthy_path = self.training_normal_images_path
        train_healthy = []
        for img_path in glob.glob(train_healthy_path):
            im = Image.open(img_path)
            train_healthy.append(np.squeeze(np.asarray(im)) / 255.0)
        len_train_healthy = len(train_healthy)
        train_healthy = np.asarray(train_healthy)

        #  INFERENCE FOR THE DATASETS
        for dataset_key in self.test_data_dict.keys():
            input_images = []
            restored_images = []
            dataset = self.test_data_dict[dataset_key]
            logging.info(f"#################################### {dataset_key} ############################################")

            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                x = data0.to(self.device)

                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)

                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0].cpu().detach().numpy()
                    x_rec_i = x_rec[i][0].cpu().detach().numpy()
                    input_images.append(x_i)
                    restored_images.append(x_rec_i)

            len_test_ = len(input_images)
            batch_size = len_test_

            nr_repeat = int(len_train_healthy / len_test_) + 1
            repeated_input = copy.deepcopy(input_images)
            repeated_input = np.repeat(np.asarray(repeated_input), nr_repeat, axis=0)
            repeated_input = repeated_input[:len_train_healthy, :, :]
            repeated_restore = copy.deepcopy(restored_images)
            repeated_restore = np.repeat(np.asarray(repeated_restore), nr_repeat, axis=0)
            repeated_restore = repeated_restore[:len_train_healthy, :, :]

            fid_PN = fid.calculate_fid_given_images([repeated_input, train_healthy],
                                                    batch_size, device, dims, num_workers)

            fid_RPN = fid.calculate_fid_given_images([repeated_restore, train_healthy],
                                                    batch_size, device, dims, num_workers)

            logging.info('[ {} ]: FID_P;N: {}'.format(dataset_key, fid_PN))
            logging.info('[ {} ]: FID_RP;N: {}'.format(dataset_key, fid_RPN))

    def normative_eval_iii_CACI(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ Normative Evaluation: Req iii.  (CACI) #################")
        lpips_alex = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True,
                                     lpips=True).to(self.device) # best forward scores
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MSE': [],
            'MSE_an': [],
            'LPIPS': [],
            'LPIPS_an': [],
            'SSIM': [],
            'SSIM_an': [],
        }
        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]
            logging.info(f"#################################### {dataset_key} ############################################")

            test_metrics = {
                'MSE': [],
                'MSE_an': [],
                'LPIPS': [],
                'LPIPS_an': [],
                'SSIM': [],
                'SSIM_an': [],
            }
            logging.info('DATASET: {}'.format(dataset_key))
            tps, fns, fps = 0, 0, []
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                x = data0.to(self.device)
                masks_bool = True if len(data1) > 2 else False
                nr_batches, nr_slices, width, height = data0.shape
                # x = data0.view(nr_batches * nr_slices, 1, width, height)
                masks = data[1][:, 0, :, :].view(nr_batches, 1, width, height).numpy()\
                    if masks_bool else None
                neg_masks = data[1][:, 1, :, :].view(nr_batches, 1, width, height).numpy()
                neg_masks[neg_masks>0.5] = 1
                neg_masks[neg_masks<1] = 0

                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))

                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)

                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    loss_lpips = np.squeeze(lpips_alex(x, x_rec, normalize=True).cpu().detach().numpy())
                    loss_lpips_h = copy.deepcopy(loss_lpips) * np.squeeze(np.abs(neg_masks))
                    loss_lpips_h[loss_lpips_h == 0] = np.nan
                    test_metrics['LPIPS'].append(np.nanmean(loss_lpips_h))

                    loss_lpips_an = copy.deepcopy(loss_lpips) * np.squeeze(np.abs(masks))
                    loss_lpips_an[loss_lpips_an == 0] = np.nan
                    loss_lpips_an = np.nanmean(loss_lpips_an)
                    test_metrics['LPIPS_an'].append(loss_lpips_an)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    masked_mse_an = np.sum(((x_rec_ - x_) * np.squeeze(masks)) ** 2.0) / np.sum(masks)
                    test_metrics['MSE_an'].append(masked_mse_an)
                    masked_mse = np.sum(((x_rec_ - x_) * np.squeeze(np.abs(neg_masks))) ** 2.0) / np.sum(neg_masks)
                    test_metrics['MSE'].append(masked_mse)

                    ssim_val, ssim_map = ssim(x_rec_, x_, data_range=1., full=True)
                    ssim_h = copy.deepcopy(ssim_map) * np.squeeze(np.abs(neg_masks))
                    ssim_h[ssim_h == 0] = np.nan
                    test_metrics['SSIM'].append(np.nanmean(ssim_h))
                    ssim_map *= np.squeeze(masks)
                    ssim_map[ssim_map == 0] = np.nan
                    ssim_ = np.nanmean(ssim_map)
                    test_metrics['SSIM_an'].append(ssim_)


                    if (idx % 5) == 0:  # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                        to_visualize = [
                            {'title': 'x', 'tensor': x_},
                            {'title': f'x_rec {np.round(np.nanmean(ssim_h), 4)}', 'tensor': x_rec_},
                        ]

                        self._log_visualization(to_visualize, dataset_key, count)


            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})

    def thresholding(self, global_model):
        """
                Validation of downstream tasks
                Logs results to wandb

                :param global_model:
                    Global parameters
                """
        logging.info("################ Threshold Search #################")
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        ths = np.linspace(0, 1, endpoint=False, num=1000)
        fprs = dict()
        for th_ in ths:
            fprs[th_] = []
        im_scale = 128 * 128
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(x)
                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    anomaly_map_i = anomaly_map[i][0]
                    for th_ in ths:
                        fpr = (np.count_nonzero(anomaly_map_i > th_) * 100) / im_scale
                        fprs[th_].append(fpr)
                    x_combo = copy.deepcopy(anomaly_map_i)
                    x_combo[x_combo < 0.007] = 0
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(4, 4)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(np.squeeze(x_combo), cmap='plasma', vmin=0, vmax=x_combo.max())
                    fig.savefig(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_anomaly.png', dpi=300)
                    cv2.imwrite(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.png',
                                (x_rec_dict['x_rec'][i][0].cpu().detach().numpy() * 255).astype(np.uint8))
                    cv2.imwrite(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_orig.png',
                                (x[i][0].cpu().detach().numpy() * 255).astype(np.uint8))
        mean_fprs = []
        for th_ in ths:
            mean_fprs.append(np.mean(fprs[th_]))
        mean_fprs = np.asarray(mean_fprs)
        sorted_idxs = np.argsort(mean_fprs)
        th_1, th_2, best_th = 0, 0, 0
        fpr_1, fpr_2, best_fpr = 0, 0, 0
        for sort_idx in sorted_idxs:
            th_ = ths[sort_idx]
            fpr_ = mean_fprs[sort_idx]
            if fpr_ <= 1:
                th_1 = th_
                fpr_1 = fpr_
            if fpr_ <= 2:
                th_2 = th_
                fpr_2 = fpr_
            if fpr_ <= 5:
                best_th = th_
                best_fpr = fpr_
            else:
                break
        print(f'Th_1: [{th_1}]: {fpr_1} || Th_2: [{th_2}]: {fpr_2} || Th_5: [{best_th}]: {best_fpr}')
        logging.info(f'Th_1: [{th_1}]: {fpr_1} || Th_2: [{th_2}]: {fpr_2} || Th_5: [{best_th}]: {best_fpr}')
        return best_th

    def print_files(self, global_model):
        """
        Validation of downstream tasks
        Prints images to disk

        :param global_model:
            Global parameters
        """

        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()


        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]

            logging.info('DATASET: {}'.format(dataset_key))

            for idx, data in enumerate(dataset):

                # Call this to get the mask size thresholds for the dataset
                # self.find_mask_size_thresholds(dataset)

                # New per batch
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks > 0] = 1

                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                x_rec = torch.clamp(x_rec, 0, 1)

                #

                for i in range(len(x)):
                        count = str(idx * len(x) + i)
                        # Don't use images with large black artifacts:
                        # Example visualizations
                        
                        x_i = x[i][0]
                        rec_2_i = x_rec[i][0]

                        res_2_i_np = anomaly_map[i][0]

                        cv2.imwrite(self.checkpoint_path + '/AnoDDPM_' + dataset_key + '_' + str(count) + '_rec.png',
                                (rec_2_i.cpu().detach().numpy() * 255).astype(np.uint8))
                        
    def object_localization(self, global_model, th=0):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ Object Localzation TEST #################" + str(th))
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': [],
            'TP': [],
            'FP': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]
            logging.info(f"#################################### {dataset_key} ############################################")

            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': [],
                'TP': [],
                'FP': [],
                'Precision': [],
                'Recall': [],
                'F1': [],
            }
            logging.info('DATASET: {}'.format(dataset_key))
            tps, fns, fps = 0, 0, []
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                x = data0.to(self.device)
                masks_bool = True if len(data1) > 2 else False
                nr_batches, nr_slices, width, height = data0.shape
                # x = data0.view(nr_batches * nr_slices, 1, width, height)
                masks = data[1][:, 0, :, :].view(nr_batches, 1, width, height).to(self.device)\
                    if masks_bool else None
                neg_masks = data[1][:, 1, :, :].view(nr_batches, 1, width, height).to(self.device)
                neg_masks[neg_masks>0.5] = 1
                neg_masks[neg_masks<1] = 0

                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))

                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)


                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    anomaly_map_i = anomaly_map[i][0]
                    anomaly_score_i = anomaly_score[i][0]
                    mask_ = masks[i][0].cpu().detach().numpy() if masks_bool else None
                    neg_mask_ = neg_masks[i][0].cpu().detach().numpy() if masks_bool else None
                    bboxes = cv2.cvtColor(neg_mask_*255, cv2.COLOR_GRAY2RGB)
                    # thresh_gt = cv2.threshold((mask_*255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    cnts_gt = cv2.findContours((mask_*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
                    gt_box = []
                    for c_gt in cnts_gt:
                        xpos, y, w, h = cv2.boundingRect(c_gt)
                        gt_box.append([xpos, y, x+w, y+h])
                        cv2.rectangle(bboxes, (xpos, y), (xpos + w, y + h), (0, 255, 0), 1)
                    #
                    loss_mse = self.criterion_rec(x_rec_i, x_i)
                    test_metrics['MSE'].append(loss_mse.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()

                    # cv2.imwrite(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_mask.png',
                    #             (mask_ * 255).astype(np.uint8))
                    # cv2.imwrite(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_neg_mask.png',
                    #             (neg_mask_ * 255).astype(np.uint8))
                    # cv2.imwrite(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.png',
                    #             (x_rec_ * 255).astype(np.uint8))
                    # cv2.imwrite(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_orig.png',
                    #             (x_ * 255).astype(np.uint8))
                    # print(anomaly_map_i.shape)


                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_mask.npy', mask_)
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_neg_mask.npy', neg_mask_)
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.npy', x_rec_)
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_anomaly.npy',
                    #         anomaly_map_i)

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

                    x_combo = copy.deepcopy(anomaly_map_i)
                    x_combo[x_combo < th] = 0

                    # cv2.imwrite(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_anomaly.png',
                    #             (np.squeeze(x_combo) * 255 * 20).astype(np.uint8))

                    # fig = plt.figure(frameon=False)
                    # fig.set_size_inches(4, 4)
# 
                    # ax = plt.Axes(fig, [0., 0., 1., 1.])
                    # ax.set_axis_off()
                    # fig.add_axes(ax)

                    # ax.imshow(np.squeeze(x_combo), cmap='plasma', vmin=0, vmax=x_combo.max())
                    # fig.savefig(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_anomaly.png', dpi=300)

                    x_pos = x_combo * mask_
                    x_neg = x_combo * neg_mask_
                    res_anomaly = np.sum(x_pos)
                    res_healthy = np.sum(x_neg)
                    # print(np.sum(x_neg), np.sum(x_pos))

                    amount_anomaly = np.count_nonzero(x_pos)
                    amount_mask = np.count_nonzero(mask_)

                    tp = 1 if amount_anomaly > 0.1 * amount_mask else 0
                    tps += tp
                    fn = 1 if tp == 0 else 0
                    fns += fn

                    fp = int(res_healthy / max(res_anomaly,1)) #[i for i in ious if i < 0.1]
                    fps.append(fp)
                    precision = tp / max((tp+fp), 1)
                    test_metrics['TP'].append(tp)
                    test_metrics['FP'].append(fp)
                    test_metrics['Precision'].append(precision)
                    test_metrics['Recall'].append(tp)
                    test_metrics['F1'].append(2 * (precision * tp) / (precision + tp + 1e-8))

                    ious = [np.round(res_anomaly,2), np.round(res_healthy,2)]

                    if (idx % 1) == 0: # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                        to_visualize = [
                            {'title': 'x', 'tensor': x_},
                            {'title': 'x_rec', 'tensor': x_rec_},
                            {'title': f'Anomaly  map {anomaly_map_i.max():.3f}', 'tensor': anomaly_map_i,
                             'cmap': 'plasma', 'vmax': anomaly_map_i.max()},
                            {'title': f'Combo map {x_combo.max():.3f}', 'tensor': x_combo,
                             'cmap': 'plasma', 'vmax': x_combo.max()}
                        ]

                        if 'mask' in x_rec_dict.keys():
                            masked_input = x_rec_dict['mask'] + x
                            masked_input[masked_input > 1] = 1
                            to_visualize.append(
                                {'title': 'Rec Orig', 'tensor': x_rec_dict['x_rec_orig'], 'cmap': 'gray'})
                            to_visualize.append({'title': 'Res Orig', 'tensor': x_rec_dict['x_res'], 'cmap': 'inferno',
                                                 'vmax': x_rec_dict['x_res'].max()})
                            to_visualize.append({'title': 'Mask', 'tensor': masked_input, 'cmap': 'gray'})

                        if masks_bool:
                            to_visualize.append({'title': 'GT', 'tensor': bboxes.astype(np.int64), 'vmax': 1})
                            to_visualize.append({'title': f'{res_anomaly}, TP: {tp}', 'tensor': x_pos,
                                                 'vmax': anomaly_map_i.max(), 'cmap': 'inferno'})
                            to_visualize.append({'title': f'{res_healthy}, FP: {fp}', 'tensor': x_neg,
                                                 'vmax': anomaly_map_i.max(), 'cmap': 'inferno'})

                            self._log_visualization(to_visualize, dataset_key, count)


            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                if metric == 'TP':
                    logging.info(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
                if metric == 'FP':
                    logging.info(f'FP: {np.sum(test_metrics[metric])} missed')
                metrics[metric].append(test_metrics[metric])

            logging.info("################################################################################")

        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})