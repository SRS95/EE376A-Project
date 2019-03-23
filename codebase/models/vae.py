import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)


    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        numSamples = x.size()[0]
        
        # Calculate the KL Divergence term
        # First, find the variational posterior mean and variance
        qm, qv = self.enc.encode(x)

        # Next, note that the marginal for Z is always the standard normal
        pm = torch.zeros([numSamples, self.z_dim], dtype=torch.float)
        pv = torch.ones([numSamples, self.z_dim], dtype=torch.float)
        
        # Now we compute the KL Divergence
        # Divide by numSamples to get the average
        kl = torch.sum(ut.kl_normal(qm, qv, pm, pv)) / numSamples

        # Approximate the reconstruction term
        # First, sample from the variational posterior
        zSample = ut.sample_gaussian(qm, qv)

        # Next, we pass the sample through the decoder to get
        # parameters for the pixel Bernoullis
        bernoulliParams = self.dec.decode(zSample)

        # Now create the approximation
        logProbForEachSample = ut.log_bernoulli_with_logits(x, bernoulliParams)
        rec = -1 * torch.sum(logProbForEachSample) / numSamples
  
        # nelbo is just kl + rec
        nelbo = kl + rec
        ################################################################################

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = {'train_loss': nelbo.item(), 'kl': kl.item(), 'rec': rec.item()}
        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
