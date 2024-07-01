import matplotlib.pyplot as plt
import numpy as np
import torch
# import umap.umap_ as umap
import seaborn as sns
import logging
import wandb


def visualize_slice(img, max=1):
    """
    Normalize for visualization
    """
    # img = img[0][0]
    img[0][0] = 0
    img[0][1] = max
    return img


def slice_3d_volume(volume, slice_id, axis):
    if axis == 0:
        return volume[slice_id, :, :]
    elif axis == 1:
        return volume[:,  slice_id, :]
    else:
        return volume[:, :, slice_id]


def vis_3d_reconstruction(img, rec, slice_id=0, prior=None, gt=None):
    axis = ['Coronal', 'Sagittal', 'Axial']
    figs = []
    for i_ax, ax in enumerate(axis):
        img_slice = slice_3d_volume(img, slice_id, i_ax).T
        max_value = np.max(img_slice)
        prior_slice = slice_3d_volume(prior, slice_id, i_ax).T if prior is not None else None
        rec_slice = slice_3d_volume(rec, slice_id, i_ax).T
        elements = [img_slice, prior_slice, rec_slice, np.abs(prior_slice - img_slice), np.abs(rec_slice-img_slice)] \
            if prior is not None else [img_slice, rec_slice, np.abs(rec_slice-img_slice)]

        if gt is not None:
            elements.append(slice_3d_volume(gt, slice_id, i_ax).T)

        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        diffp.set_size_inches(len(elements) * 4, 4)

        for i in range(len(axarr)):
            axarr[i].axis('off')
            c_map = 'gray' if i < np.ceil(len(elements)/2) else 'inferno'
            max = max_value if i < np.ceil(len(elements)/2) else 0.5
            # print(f'max:  {max}')
            axarr[i].imshow(visualize_slice(elements[i], max), cmap=c_map, origin='lower')
        figs.append(diffp)
    return figs


def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", fontsize=30, color=(0.85, 0.27, 0.41, 0.75), linewidth=1.0):
    """disp shape (2, H, W)
            code from: https://github.com/qiuhuaqi/midir
    """
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": linewidth, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return ax


# def umap_plot(self, global_model):
#     """
#     Validation on all clients after a number of rounds
#     Logs results to wandb
#
#     :param global_model:
#         Global parameters
#     """
#     logging.info("################ UMAP PLOT #################")
#     self.model.load_state_dict(global_model)
#     self.model.eval()
#     z = None
#     labels = []
#     for dataset_key in self.test_data_dict.keys():
#         dataset = self.test_data_dict[dataset_key]
#         logging.info('DATASET: {}'.format(dataset_key))
#         for idx, data in enumerate(dataset):
#             nr_batches, nr_slices, width, height = data[0].shape
#             x = data[0].view(nr_batches * nr_slices, 1, width, height)
#             x = x.to(self.device)
#             x_rec, x_rec_dict = self.model(x)
#             z = torch.flatten(x_rec_dict['z'], start_dim=1).cpu().detach().numpy() if z is None else \
#                 np.concatenate((z, torch.flatten(x_rec_dict['z'], start_dim=1).cpu().detach().numpy()), axis=0)
#             for i in range(len(x_rec_dict['z'])):
#                 labels.append(dataset_key)
#     z = np.asarray(z)
#     labels = np.asarray(labels)
#     reducer = umap.UMAP(min_dist=0.6, n_neighbors=25, metric='euclidean', init='random')
#     umap_dim = reducer.fit_transform(z)
#     sns.set_style("whitegrid")
#
#     # sns.set(context='paper', style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)
#     colors = ['#ec7062', '#af7ac4', '#5599c7', '#48c9b0', '#f5b041']
#     # fig = plt.figure()
#     # sns.color_palette(colors)
#     fig, ax = plt.subplots()
#     sns_plot = sns.jointplot(x=umap_dim[:, 1], y=umap_dim[:, 0], hue=labels, palette=sns.color_palette(colors), s=40)
#     sns_plot.ax_joint.legend(loc='center right', bbox_to_anchor=(-0.2, 0.5))
#     sns_plot.savefig(self.checkpoint_path + "/output_umap.pdf")
#     wandb.log({"umap_plot": fig})
#     # wandb.log({"umap_image": [wandb.Image(sns_plot, caption="UMAP_Image")]})
#
#     # sns_plot = sns.jointplot(x=tsne[:,1], y=tsne[:,0], hue=labels, palette="deep", s=50)
