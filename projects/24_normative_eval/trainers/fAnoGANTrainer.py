from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
import matplotlib.pyplot as plt
import copy


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        lr = training_params['optimizer_params']['lr']
        self.optimizer_E = torch.optim.Adam(self.model.encoder.parameters(),
                                            lr=lr, betas=(0.5, 0.999))
        self.kappa = 1.0

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
        self.early_stop = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, batch_loss_pl, count_images = 1.0, 1.0, 0

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h = images.shape
                count_images += b

                # Forward Pass
                self.optimizer_E.zero_grad()

                # Generate a batch of latent variables
                z = self.model.encoder(transformed_images)

                # Generate a batch of images
                fake_imgs = self.model.wgan.generator(z)

                # Real features
                real_features = self.model.wgan.discriminator.forward_features(transformed_images)
                # Fake features
                fake_features = self.model.wgan.discriminator.forward_features(fake_imgs)

                # izif architecture
                loss_imgs = self.criterion_rec(fake_imgs, transformed_images, dict())
                loss_features = self.criterion_rec(fake_features, real_features)
                loss = loss_imgs + self.kappa * loss_features

                loss.backward()
                self.optimizer_E.step()

                loss_pl = self.criterion_PL(fake_imgs, transformed_images)
                reconstructed_images = fake_imgs

                batch_loss += loss.item() * images.size(0)
                batch_loss_pl += loss_pl.item() * images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_loss_pl = batch_loss_pl / count_images if count_images > 0 else batch_loss_pl
            epoch_losses.append(epoch_loss)
            epoch_losses_pl.append(epoch_loss_pl)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_PL_": epoch_loss_pl, '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, self.client_path + '/latest_model.pt')

            img = transformed_images[0].cpu().detach().numpy()
            # print(np.min(img), np.max(img))
            rec = reconstructed_images[0].cpu().detach().numpy()
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
        }
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                x_, x_rec = self.test_model(x)
                loss_rec = self.criterion_rec(x_, x, x_rec)
                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

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
