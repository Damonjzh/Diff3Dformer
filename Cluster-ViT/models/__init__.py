# from .mortality_ViT import build_mortality_ViT
#
#
# def build_model(args):
#     return build_mortality_ViT(args)
from .mortality_ViT_DMIB_new import build_mortality_ViT
from .mortality_ViT_DMIB_new import build_mortality_ViT_new


def build_model(args):
    return build_mortality_ViT(args)

def build_model_new(args):
    return build_mortality_ViT_new(args)