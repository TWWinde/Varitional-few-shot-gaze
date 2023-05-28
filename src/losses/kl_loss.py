import torch


class KLDivergenceLoss(object):

    def __call__(self, input_dict, output_dict):
        kl_loss = -0.5 * torch.sum(1 + output_dict['logvar'] - output_dict['mu'].pow(2) - output_dict['logvar'].exp(),dim=1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

# use case
# kl_loss = KLDivergenceLoss()
# loss = kl_loss(mu, logvar)
