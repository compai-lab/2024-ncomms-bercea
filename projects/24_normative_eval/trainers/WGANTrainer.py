from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
import matplotlib.pyplot as plt
import copy
import torch.autograd as autograd


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        lr = training_params['optimizer_params']['lr']

        self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(),
                                       lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(),
                                       lr=lr, betas=(0.5, 0.999))
        self.lambda_gp = 10
        self.n_critic = 5

    def compute_gradient_penalty(self, D, real_samples, fake_samples, device):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(*real_samples.shape[:2], 1, 1, device=device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        d_interpolates = D(interpolates)
        fake = torch.ones(*d_interpolates.shape, device=device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


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
        epoch_losses_pl = []
        epoch_losses_adv = []
        epoch_losses_disc = []
        self.early_stop = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, batch_loss_pl, batch_loss_adv, batch_loss_disc, count_images = 0.0, 0.0, 0.0, 0.0, 0

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h = images.shape
                count_images += b

                # Configure input
                real_imgs = transformed_images

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.randn(transformed_images.shape[0], self.model.latent_dim, device=self.device)

                # Generate a batch of images
                fake_imgs = self.model.generator(z)

                # Real images
                real_validity = self.model.discriminator(real_imgs)
                # Fake images
                fake_validity = self.model.discriminator(fake_imgs.detach())
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.model.discriminator,
                                                            real_imgs.data,
                                                            fake_imgs.data,
                                                            self.device)
                # Adversarial loss
                d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity)
                          + self.lambda_gp * gradient_penalty)

                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                # Train the generator and output log every n_critic steps
                if (int(count_images/b)-1) % self.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = self.model.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.model.discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()

                # Reconstruction Loss
                loss_rec = self.criterion_rec(fake_imgs, transformed_images, dict())
                # loss_rec += 1 - ssim(reconstructed_images, images, data_range=1.0, size_average=True) # return a scalar

                loss_pl = self.criterion_PL(fake_imgs, transformed_images)

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # to avoid nan loss
                batch_loss += loss_rec.item() * images.size(0)
                batch_loss_pl += loss_pl.item() * images.size(0)
                batch_loss_adv += g_loss.item() * images.size(0)
                batch_loss_disc += d_loss.item() * images.size(0)
            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_loss_pl = batch_loss_pl / count_images if count_images > 0 else batch_loss_pl
            epoch_loss_adv = batch_loss_adv / count_images if count_images > 0 else batch_loss_adv
            epoch_loss_disc = batch_loss_disc / count_images if count_images > 0 else batch_loss_disc

            epoch_losses.append(epoch_loss)
            epoch_losses_pl.append(epoch_loss_pl)
            epoch_losses_adv.append(epoch_loss_adv)
            epoch_losses_disc.append(epoch_loss_disc)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_PL_": epoch_loss_pl, '_step_': epoch})
            wandb.log({"Train/Loss_ADV_": epoch_loss_adv, '_step_': epoch})
            wandb.log({"Train/Loss_DISC_": epoch_loss_disc, '_step_': epoch})


            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, self.client_path + '/latest_model.pt')

            img = transformed_images[0].cpu().detach().numpy()
            # print(np.min(img), np.max(img))
            rec = fake_imgs[0].cpu().detach().numpy()
            # print(f'rec: {np.min(rec)}, {np.max(rec)}')
            elements = [img, rec, np.abs(rec - img)]
            v_maxs = [1, 1, 0.5]
            diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            diffp.set_size_inches(len(elements) * 4, 4)
            for i in range(len(axarr)):
                axarr[i].axis('off')
                v_max = v_maxs[i]
                c_map = 'gray' if v_max == 1 else 'plasma'
                axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)

            wandb.log({'Train/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

            # Run validation
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
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
            task + '_loss_adv': 0,
        }
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                z = torch.randn(x.shape[0], self.test_model.latent_dim, device=self.device)

                # Generate a batch of images
                x_ = self.test_model.generator(z)

                fake_validity = self.model.discriminator(x_)
                g_loss = torch.mean(fake_validity)

                loss_rec = self.criterion_rec(x_, x, dict())
                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)
                metrics[task + '_loss_adv'] += g_loss.item() * x.size(0)

        img = x.detach().cpu()[0].numpy()
        rec = x_.detach().cpu()[0].numpy()

        elements = [img, rec, np.abs(rec - img)]
        v_maxs = [1, 1, 0.5]
        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        diffp.set_size_inches(len(elements) * 4, 4)
        for i in range(len(axarr)):
            axarr[i].axis('off')
            v_max = v_maxs[i]
            c_map = 'gray' if v_max == 1 else 'plasma'
            axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)

        wandb.log({task + '/Example_': [
            wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '_loss_adv'] / test_total
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
