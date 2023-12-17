from ppcls.arch.backbone.model_zoo.vision_transformer import ViT_small_patch16_224
from ppcls.arch.backbone.model_zoo.se_resnet_vd import SE_ResNet18_vd

def choose_model(options):

    model_name = str(options['model']).lower()

    if model_name == 'vit':
        return ViT_small_patch16_224(pretrained = True,
                                    img_size = 224,
                                    class_num = 10)
    elif model_name == 'resnet':
        return SE_ResNet18_vd(pretrained = True,
                            class_num = 10)

