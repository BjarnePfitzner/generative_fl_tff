from .soft_intro_vae_loss import SoftIntroVaeLossFns
from .intro_vae_loss import IntroVaeLossFns
from .kl_loss import VaeKlReconstructionLossFns


def get_loss_fn(loss_fn_cfg):
    if loss_fn_cfg.name == 'kl_loss':
        return VaeKlReconstructionLossFns(beta=loss_fn_cfg.kl_beta,
                                          beta_warmup=loss_fn_cfg.kl_beta_warmup)
    elif loss_fn_cfg.name == 'intro_vae_loss':
        return IntroVaeLossFns(alpha=loss_fn_cfg.intro_alpha,
                               alpha_warmup=loss_fn_cfg.intro_alpha_warmup,
                               beta=loss_fn_cfg.intro_beta,
                               m=loss_fn_cfg.intro_m)
    elif loss_fn_cfg.name == 'soft_intro_vae_loss':
        return SoftIntroVaeLossFns(beta_rec=loss_fn_cfg.s_intro_beta_rec,
                                   beta_kl=loss_fn_cfg.s_intro_beta_kl,
                                   beta_neg=loss_fn_cfg.s_intro_beta_neg,
                                   gamma_r=loss_fn_cfg.s_intro_gamma_r)

    raise NotImplementedError('Loss Function not defined')
