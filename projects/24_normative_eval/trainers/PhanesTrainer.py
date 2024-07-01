from core.Trainer import Trainer
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR

from time import time
import wandb
from dl_utils.config_utils import *
import logging
from model_zoo.soft_intro_vae_daniel import *
from model_zoo.aotgan.loss import loss as loss_module
from optim.losses.image_losses import EmbeddingLoss
import matplotlib.pyplot as plt
import copy
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure
from skimage.measure import label, regionprops


class PTrainer(Trainer):
    """
    Contains code from Soft-Intro VAE for image datasets
    Code based on: https://github.com/taldatech/soft-intro-vae-pytorch/blob/main/soft_intro_vae/

    T. Daniel and A. Tamar. Soft-introvae: Analyzing and improving the introspective variational autoencoder.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4391–4400, 2021.

    and AOTGAN for high-resolution image in-painting
    """
    def __init__(self, training_params, model, data, device, log_wandb=True):

        self.optimizer_e = Adam(model.encoder.parameters(), lr=training_params['optimizer_params']['lr'])
        self.optimizer_d = Adam(model.decoder.parameters(), lr=training_params['optimizer_params']['lr'])
        self.optimizer_netG = Adam(model.netG.parameters(), lr=training_params['optimizer_params']['lr'])
        self.optimizer_netD = Adam(model.netD.parameters(), lr=training_params['optimizer_params']['lr'])

        self.e_scheduler = MultiStepLR(self.optimizer_e, milestones=(100,), gamma=0.1)
        self.d_scheduler = MultiStepLR(self.optimizer_d, milestones=(100,), gamma=0.1)
        self.netG_scheduler = MultiStepLR(self.optimizer_netG, milestones=(100,), gamma=0.1)
        self.netD_scheduler = MultiStepLR(self.optimizer_netD, milestones=(100,), gamma=0.1)

        self.scale = 1 / (training_params['input_size'][1] ** 2)  # normalize by images size (channels * height * width)
        self.gamma_r = 1e-8
        self.beta_kl = 1.0
        self.beta_rec = 0.5
        self.beta_neg = 128.0
        self.z_dim = 128
        rec_loss = '1*L1+250*Style+0.1*Perceptual'
        self.adv_weight = 0.01
        gan_type = 'smgan'
        losses = list(rec_loss.split('+'))
        self.rec_loss = {}
        for l in losses:
            weight, name = l.split('*')
            self.rec_loss[name] = float(weight)
        # set up losses and metrics
        self.rec_loss_func = {
            key: getattr(loss_module, key)() for key, val in self.rec_loss.items()}
        self.adv_loss = getattr(loss_module, gan_type)()

        self.embedding_loss = EmbeddingLoss()
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)

    def filter_anomaly_mask(self, anomaly_masks):
        filtered_masks = []
        for b in range(anomaly_masks.shape[0]):
            anomaly_mask = anomaly_masks[b][0]
            anomaly_mask[anomaly_mask > 0.2] = 1
            anomaly_mask[anomaly_mask < 1] = 0
            # Compute connected components
            label_image = label(anomaly_mask)

            # Initialize an empty list to store the pixels of the connected components
            pixels = []

            # Iterate over the connected components
            for region in regionprops(label_image):
                # Get the number of pixels in the component
                num_pixels = region.area
                # print(num_pixels)
                # Only keep the component if it has more than 5 pixels
                if num_pixels > 5:
                    # Append the pixels of the component to the list
                    pixels.extend(region.coords)

            # Create an empty binary mask of the same size as the original image
            filtered_mask = np.zeros(anomaly_mask.shape)

            # Set the filtered pixels in the binary mask to 1
            for pixel in pixels:
                filtered_mask[pixel[0], pixel[1]] = 1.0

            filtered_mask = gaussian_filter(filtered_mask, sigma=1.2)
            filtered_mask[filtered_mask > 0.5] = 1.0
            filtered_mask[filtered_mask < 1] = 0

            filtered_masks.append(np.expand_dims(filtered_mask, 0))
        filtered_masks = np.asarray(filtered_masks)

        return filtered_masks

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

        epoch_losses = []
        self.early_stop = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()

            diff_kls, batch_kls_real, batch_kls_fake, batch_kls_rec, batch_rec_errs, batch_exp_elbo_f,\
            batch_exp_elbo_r, batch_emb, count_images = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            batch_netGD_rec, batch_netG_loss, batch_netD_loss = 0.0, 0.0, 0.0

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                transformed_images = copy.deepcopy(images)

                if self.transform is not None:
                    masks = self.transform(transformed_images)
                else:
                    masks = torch.zeros_like(images.shape).to(self.device)

                b, c, w, h = images.shape

                count_images += b

                noise_batch = torch.randn(size=(b, self.z_dim)).to(self.device)
                real_batch = images.to(self.device)

                # =========== Update E ================
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                for param in self.model.decoder.parameters():
                    param.requires_grad = False
                for param in self.model.netG.parameters():
                    param.requires_grad = False
                for param in self.model.netD.parameters():
                    param.requires_grad = False

                fake = self.model.sample(noise_batch)
                # cloned_encoder = copy.deepcopy(self.model.encoder)
                real_mu, real_logvar, anomaly_embeddings = self.model.encode(real_batch)
                z = reparameterize(real_mu, real_logvar)
                rec = self.model.decoder(z)

                _, _, healthy_embeddings = self.model.encode(rec.detach())

                loss_emb = self.embedding_loss(anomaly_embeddings['embeddings'], healthy_embeddings['embeddings'])

                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")
                lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")
                # loss_emb = loss_rec
                rec_rec, z_dict = self.model.ae(rec.detach(), deterministic=False)
                rec_mu, rec_logvar, z_rec = z_dict['z_mu'], z_dict['z_logvar'], z_dict['z']
                rec_fake, z_dict_fake = self.model.ae(fake.detach(), deterministic=False)
                fake_mu, fake_logvar, z_fake = z_dict_fake['z_mu'], z_dict_fake['z_logvar'], z_dict_fake['z']

                kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
                kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")

                loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type="mse", reduction='none')
                while len(loss_rec_rec_e.shape) > 1:
                    loss_rec_rec_e = loss_rec_rec_e.sum(-1)
                loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type="mse", reduction='none')
                while len(loss_rec_fake_e.shape) > 1:
                    loss_rec_fake_e = loss_rec_fake_e.sum(-1)

                expelbo_rec = (-2 * self.scale * (self.beta_rec * loss_rec_rec_e + self.beta_neg * kl_rec)).exp().mean()
                expelbo_fake = (-2 * self.scale * (self.beta_rec * loss_rec_fake_e + self.beta_neg * kl_fake)).exp().mean()

                lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
                lossE_real = self.scale * (self.beta_rec * loss_rec + self.beta_kl * lossE_real_kl)

                lossE = lossE_real + lossE_fake + 0.005 * loss_emb
                self.optimizer_e.zero_grad()
                lossE.backward()
                self.optimizer_e.step()

                # ========= Update D ==================
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.netG.parameters():
                    param.requires_grad = False
                for param in self.model.netD.parameters():
                    param.requires_grad = False

                fake = self.model.sample(noise_batch)
                rec = self.model.decoder(z.detach())
                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")

                rec_mu, rec_logvar,_ = self.model.encode(rec)
                z_rec = reparameterize(rec_mu, rec_logvar)

                fake_mu, fake_logvar,_ = self.model.encode(fake)
                z_fake = reparameterize(fake_mu, fake_logvar)

                rec_rec = self.model.decode(z_rec.detach())
                rec_fake = self.model.decode(z_fake.detach())

                loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type="mse", reduction="mean")
                loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type="mse", reduction="mean")

                lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
                lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")


                lossD = self.scale * (loss_rec * self.beta_rec + (
                        lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl + self.gamma_r * 0.5 * self.beta_rec * (
                                         loss_rec_rec + loss_fake_rec))

                self.optimizer_d.zero_grad()
                lossD.backward()
                self.optimizer_d.step()
                if torch.isnan(lossD) or torch.isnan(lossE):
                    print('is non for D')
                    raise SystemError
                if torch.isnan(lossE):
                    print('is non for E')
                    raise SystemError

                diff_kls += -lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item() * images.shape[0]
                batch_kls_real += lossE_real_kl.data.cpu().item() * images.shape[0]
                batch_kls_fake += lossD_fake_kl.cpu().item() * images.shape[0]
                batch_kls_rec += lossD_rec_kl.data.cpu().item() * images.shape[0]
                batch_rec_errs += loss_rec.data.cpu().item() * images.shape[0]

                batch_exp_elbo_f += expelbo_fake.data.cpu() * images.shape[0]
                batch_exp_elbo_r += expelbo_rec.data.cpu() * images.shape[0]

                batch_emb += loss_emb.cpu().item() * images.shape[0]

                # ========= Update PH Generator  ==================
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = False
                for param in self.model.netG.parameters():
                    param.requires_grad = True
                for param in self.model.netD.parameters():
                    param.requires_grad = True

                x_res = np.abs(exposure.equalize_adapthist(rec.cpu().detach().numpy()) - exposure.equalize_adapthist(images.cpu().detach().numpy()))

                ano_mask = torch.Tensor(self.filter_anomaly_mask(x_res)).to(self.device)
                mixed_mask = ano_mask + masks
                mixed_mask[mixed_mask > 1] = 1

                transformed_images = (images * (1 - mixed_mask).float()) + mixed_mask

                pred_img = self.model.netG(transformed_images, mixed_mask)
                comp_img = (1 - mixed_mask) * images + mixed_mask * pred_img

                # reconstruction losses
                losses = {}
                for name, weight in self.rec_loss.items():
                    losses[name] = weight * self.rec_loss_func[name](pred_img, images)

                # adversarial loss
                dis_loss, gen_loss = self.adv_loss(self.model.netD, comp_img, images, masks)
                loss_advg = gen_loss * self.adv_weight

                # backforward
                self.optimizer_netG.zero_grad()
                self.optimizer_netD.zero_grad()
                sum(losses.values()).backward()
                dis_loss.backward()
                self.optimizer_netG.step()
                self.optimizer_netD.step()

                batch_netGD_rec += sum(losses.values()).cpu().item() * images.shape[0]
                batch_netG_loss += loss_advg.cpu().item() * images.shape[0]
                batch_netD_loss += dis_loss.cpu().item() * images.shape[0]


            epoch_loss_d_kls = diff_kls / count_images if count_images > 0 else diff_kls
            epoch_loss_kls_real = batch_kls_real / count_images if count_images > 0 else batch_kls_real
            epoch_loss_kls_fake = batch_kls_fake / count_images if count_images > 0 else batch_kls_fake
            epoch_loss_kls_rec = batch_kls_rec / count_images if count_images > 0 else batch_kls_rec
            epoch_loss_rec_errs = batch_rec_errs / count_images if count_images > 0 else batch_rec_errs
            epoch_loss_exp_f = batch_exp_elbo_f / count_images if count_images > 0 else batch_exp_elbo_f
            epoch_loss_exp_r = batch_exp_elbo_r / count_images if count_images > 0 else batch_exp_elbo_r
            epoch_loss_emb = batch_emb / count_images if count_images > 0 else batch_emb
            epoch_loss_netGD_rec = batch_netGD_rec / count_images if count_images > 0 else batch_netGD_rec
            epoch_loss_netG_loss = batch_netG_loss / count_images if count_images > 0 else batch_netG_loss
            epoch_loss_netD_loss = batch_netD_loss / count_images if count_images > 0 else batch_netD_loss

            epoch_losses.append(epoch_loss_rec_errs)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss_rec_errs, end_time - start_time, count_images))
            wandb.log({"Train/Loss_DKLS": epoch_loss_d_kls, '_step_': epoch})
            wandb.log({"Train/Loss_REAL": epoch_loss_kls_real, '_step_': epoch})
            wandb.log({"Train/Loss_FAKE": epoch_loss_kls_fake, '_step_': epoch})
            wandb.log({"Train/Loss_REC": epoch_loss_kls_rec, '_step_': epoch})
            wandb.log({"Train/Loss_REC_ERRS": epoch_loss_rec_errs, '_step_': epoch})
            wandb.log({"Train/Loss_EXP_F": epoch_loss_exp_f, '_step_': epoch})
            wandb.log({"Train/Loss_EXP_R": epoch_loss_exp_r, '_step_': epoch})
            wandb.log({"Train/Loss_EMB": epoch_loss_emb, '_step_': epoch})
            wandb.log({"Train/Loss_netGD_REC": epoch_loss_netGD_rec, '_step_': epoch})
            wandb.log({"Train/Loss_netG": epoch_loss_netG_loss, '_step_': epoch})
            wandb.log({"Train/Loss_netD": epoch_loss_netD_loss, '_step_': epoch})


            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, self.client_path + '/latest_model.pt')


            img = images[0].cpu().detach().numpy()
            ano_mask_ = mixed_mask[0].cpu().detach().numpy()
            mask_ = masks[0].cpu().detach().numpy()
            # print(np.min(img), np.max(img))
            rec_ = rec[0].cpu().detach().numpy()
            trans_ = transformed_images[0].cpu().detach().numpy()
            final_rec_ = comp_img[0].cpu().detach().numpy()
            # print(f'rec: {np.min(rec)}, {np.max(rec)}')
            x_res_coarse = np.abs(exposure.equalize_adapthist(rec_) - exposure.equalize_adapthist(img))
            x_res = np.abs(exposure.equalize_adapthist(final_rec_) - exposure.equalize_adapthist(img))

            elements = [img, rec_, x_res_coarse, mask_, ano_mask_, trans_, final_rec_, img, x_res]
            v_maxs = [1, 1, 0.5, 0.99, 0.99, 1, 1, 1, 0.5]
            titles = ['Input', 'Rec (LR)', str(np.sum(x_res_coarse)), 'Mask (Synth)', 'Mask (Ano)', 'Masked', 'Rec (HR)', 'Input', str(np.sum(x_res))]

            diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            diffp.set_size_inches(len(elements) * 4, 4)
            for i in range(len(axarr)):
                axarr[i].axis('off')
                v_max = v_maxs[i]
                c_map = 'gray' if v_max == 1 else 'inferno'
                axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)
                axarr[i].set_title(titles[i])

            wandb.log({'Train/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})


            self.test(self.model.state_dict(), self.val_ds, 'Val', [self.optimizer_e.state_dict(),
                                                                    self.optimizer_d.state_dict()], epoch)

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
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
            task + '_loss_mse_coarse': 0,
            task + '_loss_pl_coarse': 0,
        }
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                x_, z_rec = self.test_model(x)
                x_coarse = z_rec['y_coarse']
                loss_rec = (self.criterion_MSE(x_, x) + self.criterion_MSE(x_coarse, x)) / 2
                loss_mse = self.criterion_MSE(x_, x)
                loss_mse_coarse = self.criterion_MSE(x_coarse, x)
                loss_pl = self.criterion_PL(x_, x)
                loss_pl_coarse = self.criterion_PL(x_coarse, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)
                metrics[task + '_loss_mse_coarse'] += loss_mse_coarse.item() * x.size(0)
                metrics[task + '_loss_pl_coarse'] += loss_pl_coarse.item() * x.size(0)

        img = x[0].cpu().detach().numpy()
        # print(np.min(img), np.max(img))
        rec_coarse = x_coarse[0].cpu().detach().numpy()
        masked_x = z_rec['masked'][0].cpu().detach().numpy()
        rec_ = x_[0].cpu().detach().numpy()
        # print(f'rec: {np.min(rec)}, {np.max(rec)}')
        x_res_coarse = np.abs(exposure.equalize_adapthist(rec_coarse) - exposure.equalize_adapthist(img))
        x_res = np.abs(exposure.equalize_adapthist(rec_) - exposure.equalize_adapthist(img))
        elements = [img, rec_coarse, x_res_coarse, masked_x, rec_, img, x_res]
        v_maxs = [1, 1, 0.5, 1, 1, 1, 0.5]
        titles = ['Input', 'Rec (LR)', str(np.sum(x_res_coarse)), 'Masked',  'Rec (HR)', 'Input', str(np.sum(x_res))]
        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        diffp.set_size_inches(len(elements) * 4, 4)
        for i in range(len(axarr)):
            axarr[i].axis('off')
            v_max = v_maxs[i]
            c_map = 'gray' if v_max == 1 else 'inferno'
            axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)
            axarr[i].set_title(titles[i])

        wandb.log({task + '/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer_e.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '_loss_rec'] / test_total
        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                self.best_weights = model_weights
                self.best_opt_weights = opt_weights
                torch.save({'model_weights': model_weights, 'optimizer_e_weights': opt_weights[0],
                            'optimizer_d_weights': opt_weights[1], 'epoch': epoch},
                           self.client_path + '/best_model.pt')
            self.early_stop = self.early_stopping(epoch_val_loss)
            self.e_scheduler.step(epoch_val_loss)
            self.d_scheduler.step(epoch_val_loss)
