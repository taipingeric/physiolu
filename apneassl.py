import torch
from torch import nn
from torch.nn import functional as F

from pl_bolts.models.self_supervised import SimCLR, SimSiam, BYOL
from pl_bolts.models.self_supervised.simsiam.models import MLP

from copy import deepcopy

class ECGBYOL(BYOL):
    def __init__(self, 
                 encoder, 
                 encoder_out_dim,
                 projector_hidden_size=4096,
                 projector_out_dim=256):
        super().__init__(num_classes=1, base_encoder=encoder, encoder_out_dim=encoder_out_dim)
        print(encoder_out_dim,
              projector_hidden_size,
              projector_out_dim)
        self.online_network = SiameseArm(encoder, 
                                         encoder_out_dim, 
                                         projector_hidden_size, 
                                         projector_out_dim)
        self.target_network = deepcopy(self.online_network)
    
    def shared_step(self, batch, batch_idx):
        img_1, img_2 = batch
#         img_1, img_2 = imgs[:2]
    
        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = -2 * F.cosine_similarity(h1, z2).mean()

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        # L2 normalize
        loss_b = -2 * F.cosine_similarity(h1, z2).mean()

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss
    

class SiameseArm(nn.Module):
    def __init__(
        self,
        encoder,
        input_dim = 2048,
        hidden_size = 4096,
        output_dim = 256,
    ):
        super().__init__()
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(input_dim, hidden_size, output_dim)
        # Predictor
        self.predictor = MLP(output_dim, hidden_size, output_dim)

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h
    
class ECGSimSiam(SimSiam):
    def __init__(self, encoder, config, **kargs):
        print(kargs)
        super().__init__(**kargs)
        self.encoder = encoder
        self.online_network = SiameseArm(
            encoder, input_dim=config.dim, hidden_size=self.hidden_mlp, output_dim=self.feat_dim
        )
        
    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y
    def training_step(self, batch, batch_idx):
        img_1, img_2 = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        # log results
        self.log_dict({"train_loss": loss})

        return loss