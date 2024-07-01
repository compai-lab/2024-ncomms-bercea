import torch
import numpy as np
from net_utils.initialize import *
from model_zoo.vqvae import VQVAE
from net_utils.inferers.inferer import VQVAETransformerInferer
from model_zoo.transformers import DecoderOnlyTransformer
from net_utils.ordering import OrderingType, Ordering
import torch.nn.functional as F

class LTM(nn.Module):
    def __init__(self, path_vqvae_checkpoints: str, path_trans_checkpoints: str):
        """
        :param path_predictions:
            Path to numpy predctions of desired model
        :param in_channels:
        """
        device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

        super(LTM, self).__init__()

        self.inferer = VQVAETransformerInferer()
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
        vqvae_checkpoint = torch.load(path_vqvae_checkpoints, map_location=torch.device(device))['model_weights']
        self.vqvae.load_state_dict(vqvae_checkpoint)
        test_shape = torch.zeros(16, 64, 32, 32)
        self.ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + test_shape.shape[2:])

        self.transformer = DecoderOnlyTransformer(
            num_tokens=17,
            max_seq_len=1024,
            attn_layers_dim=128,
            attn_layers_depth=16,
            attn_layers_heads=16
        )
        self.transformer.to(device)
        trans_checkpoint = torch.load(path_trans_checkpoints, map_location=torch.device(device))['model_weights']
        self.transformer.load_state_dict(trans_checkpoint)

    def forward(self, x):
        log_likelihood = self.inferer.get_likelihood(
            inputs=x,
            vqvae_model=self.vqvae,
            transformer_model=self.transformer,
            ordering=self.ordering,
        )
        likelihood = torch.exp(log_likelihood)
        mask = log_likelihood.cpu()[0, ...] < torch.quantile(log_likelihood, 0.03).item()
        mask_flattened = mask.reshape(-1)
        mask_flattened = mask_flattened[self.ordering.get_sequence_ordering()]

        latent = self.vqvae.index_quantize(x)
        latent = latent.reshape(latent.shape[0], -1)
        latent = latent[:, self.ordering.get_sequence_ordering()]
        latent = F.pad(latent, (1, 0), "constant", self.vqvae.num_embeddings)
        latent = latent.long()
        latent_healed = latent.clone()

        # heal the sequence
        # loop over tokens
        for i in range(1, latent.shape[1]):
            if mask_flattened[i - 1]:
                # if token is low probability, replace with tranformer's most likely token
                logits = self.transformer(latent_healed[:, :i])
                probs = F.softmax(logits, dim=-1)
                # don't sample beginning of sequence token
                probs[:, :, self.vqvae.num_embeddings] = 0
                index = torch.argmax(probs[0, -1, :])
                latent_healed[:, i] = index

        # reconstruct
        latent_healed = latent_healed[:, 1:]
        latent_healed = latent_healed[:, self.ordering.get_revert_sequence_ordering()]
        latent_healed = latent_healed.reshape((32, 32))

        image_healed = self.vqvae.decode_samples(latent_healed[None, ...])

        return image_healed

    def get_anomaly(self, x):
        # print(file_names)
        x_rec = self.forward(x)
        x_rec_ = x_rec.cpu().detach().numpy()
        x_ = x.cpu().detach().numpy()
        anomaly_maps=np.abs(x_-x_rec_)
        anomaly_score = np.mean(anomaly_maps, axis=(1 ,2, 3), keepdims=True)
        return anomaly_maps, anomaly_score, {'x_rec': x_rec}
