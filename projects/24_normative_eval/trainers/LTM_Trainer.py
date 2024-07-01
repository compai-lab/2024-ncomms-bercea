from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
import matplotlib.pyplot as plt
import copy
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from net_utils.inferers.inferer import VQVAETransformerInferer
from torch.nn import CrossEntropyLoss, L1Loss
from model_zoo.vqvae import VQVAE
from net_utils.ordering import OrderingType, Ordering

"""
!!!! SET CHECKPOINT PATH AFTER VQVAE TRAINING  !!!!
"""
class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        self.inferer = VQVAETransformerInferer()
        self.ce_loss = CrossEntropyLoss()
        self.vqvae = VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_channels=(256, 256),
            num_res_channels=(256, 256),
            num_embeddings=16,
            embedding_dim=64)
        self.vqvae.to(device)
        checkpoint_path = './weights/23_uad_review/ltm/2023_11_06_14_34_01_814383/best_model.pt'
        vqvae_checkpoint = torch.load( checkpoint_path, map_location=torch.device(device))['model_weights']
        self.vqvae.load_state_dict(vqvae_checkpoint)
        test_shape = torch.zeros(16, 64, 32, 32)
        self.ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + test_shape.shape[2:])



        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        if model_state is not None:
            self.model.load_state_dict(model_state)  # load weights
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        epoch_losses = []
        epoch_losses_q = []
        self.early_stop = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, batch_loss_q, count_images = 1.0, 1.0, 0

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h = images.shape
                count_images += b

                # Forward Pass
                self.optimizer.zero_grad()

                logits, target, _ = self.inferer(images, self.vqvae, self.model, self.ordering, return_latent=True)
                logits = logits.transpose(1, 2)

                loss = self.ce_loss(logits, target)

                loss.backward()
                self.optimizer.step()
                # reconstructed_images, _ = self.vqvae(images)
                # reconstructed_images = self.inferer.sample(
                #     vqvae_model=self.vqvae,
                #     transformer_model=self.model,
                #     ordering=self.ordering,
                #     latent_spatial_dim=(16, 64),
                #     starting_tokens=torch.Tensor(self.vqvae.num_embeddings).to(self.device) * torch.ones((1, 1), device=self.device),
                # )

                batch_loss += loss.item() * images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, self.client_path + '/latest_model.pt')

            # img = transformed_images[0].cpu().detach().numpy()
            # # print(np.min(img), np.max(img))
            # rec = reconstructed_images[0].cpu().detach().numpy()
            # # print(f'rec: {np.min(rec)}, {np.max(rec)}')
            # elements = [img, rec, np.abs(rec - img)]
            # v_maxs = [1, 1, 0.5]
            # diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            # diffp.set_size_inches(len(elements) * 4, 4)
            # for i in range(len(axarr)):
            #     axarr[i].axis('off')
            #     v_max = v_maxs[i]
            #     c_map = 'gray' if v_max == 1 else 'inferno'
            #     axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)
            #
            # wandb.log({'Train/Example_': [
            #     wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

            # Run validation
            # if epoch % 50 == 0:
            self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + '_loss_rec': 0
            # ,task + '_loss_mse': 0
            # ,task + '_loss_pl': 0
        }
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                logits, target, _ = self.inferer(x, self.vqvae, self.test_model, self.ordering, return_latent=True)
                logits = logits.transpose(1, 2)

                loss = self.ce_loss(logits, target)

                # x_ = self.inferer.sample(
                #     vqvae_model=self.vqvae,
                #     transformer_model=self.test_model,
                #     ordering=self.ordering,
                #     latent_spatial_dim=(32, 32),
                #     starting_tokens=torch.Tensor(self.vqvae.num_embeddings).to(self.device) * torch.ones((1, 1), device=self.device),
                # )

                loss_rec = loss
                # loss_mse = self.criterion_MSE(x_, x)
                # loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                # metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                # metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

        # img = x.detach().cpu()[0].numpy()
        # rec = x_.detach().cpu()[0].numpy()

        # elements = [img, rec, np.abs(rec - img)]
        # v_maxs = [1, 1, 0.5]
        # diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        # diffp.set_size_inches(len(elements) * 4, 4)
        # for i in range(len(axarr)):
        #     axarr[i].axis('off')
        #     v_max = v_maxs[i]
        #     c_map = 'gray' if v_max == 1 else 'inferno'
        #     axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)
        #
        # wandb.log({task + '/Example_': [
        #     wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '_loss_rec'] / test_total
        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                           self.client_path + '/best_model.pt')
                self.best_weights = copy.deepcopy(model_weights)
                self.best_opt_weights = copy.deepcopy(opt_weights)
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)
