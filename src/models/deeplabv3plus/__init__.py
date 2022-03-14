# from models import MODEL

class Configuration():
    def __init__(self):
        self.MODEL_NAME = 'deeplabv3plus'
        self.MODEL_BACKBONE = 'xception'
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SHORTCUT_KERNEL = 1
        self.MODEL_NUM_CLASSES = 8
        self.MODEL_AUX_OUT = 4
        self.TRAIN_BN_MOM = 0.0003


def get_deeplabv3plus_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from models.deeplabv3plus.deeplabv3plus import deeplabv3plus, deeplabv3plus3branch, deeplabv3plus3branch2, \
        deeplabv3pluslabel, deeplabv3plus3branch_1, deeplabv3plus_1, deeplabv3plus3branchsegattention, \
        deeplabv3pluswithbboxandseg, deeplabv3pluscatbboxandseg
    from models.deeplabv3plus.deeplabv3plusselflearning import deeplabv3plus as deeplabv3plus_seg_selflearning
    if name.lower() == 'deeplabv3plusxception-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxception' or name.lower() == 'deeplabv3plusxception-16os':
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxception_1':
        return deeplabv3plus_1(cfg)
    elif name.lower() == 'deeplabv3plusxceptiondensity':
        cfg.MODEL_AUX_OUT = 6
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxceptiondensityx':
        cfg.MODEL_AUX_OUT = 5
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxception5':
        cfg.MODEL_AUX_OUT = 5
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3pluswithbboxandseg':
        return deeplabv3pluswithbboxandseg(cfg)
    elif name.lower() == 'deeplabv3pluscatbboxandseg':
        return deeplabv3pluscatbboxandseg(cfg)
    elif name.lower() == 'deeplabv3pluscatbboxandsegms':
        return deeplabv3pluscatbboxandseg(cfg, multi=True)

    elif name.lower() == 'deeplabv3plus3branch':
        return deeplabv3plus3branch(cfg)
    elif name.lower() == 'deeplabv3plus3branchsegattention':
        return deeplabv3plus3branchsegattention(cfg)

    elif name.lower() == 'deeplabv3plus3branch1.1':
        return deeplabv3plus3branch_1(cfg)
    elif name.lower() == 'deeplabv3plus3branchdensity':
        cfg.MODEL_AUX_OUT = 6
        return deeplabv3plus3branch(cfg)
    elif name.lower() == 'deeplabv3plus3branchdensityx':
        cfg.MODEL_AUX_OUT = 5
        return deeplabv3plus3branch(cfg)

    elif name.lower() == 'deeplabv3plus3branch2':
        return deeplabv3plus3branch2(cfg)

    elif name.lower() == 'deeplabv3pluslabel':
        return deeplabv3pluslabel(cfg)

    elif name.lower() == 'deeplabv3plusxceptionthick' or name.lower() == 'deeplabv3plusthick':
        cfg.MODEL_ASPP_OUTDIM = 512
        cfg.MODEL_SHORTCUT_DIM = 96
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnet101-8os':
        cfg.MODEL_BACKBONE = 'res101_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnet101' or name.lower() == 'deeplabv3plusatrousresnet101-16os':
        cfg.MODEL_BACKBONE = 'res101_atrous'
        return deeplabv3plus(cfg)

    elif name.lower() == 'deeplabv3plusatrousresnext101-8os':
        cfg.MODEL_BACKBONE = 'resnext101_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnext101' or name.lower() == 'deeplabv3plusatrousresnext101-16os':
        cfg.MODEL_BACKBONE = 'resnext101_atrous'
        return deeplabv3plus(cfg)

    elif name.lower() == 'deeplabv3plusatrousresnet152-8os':
        cfg.MODEL_BACKBONE = 'res152_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnet152' or name.lower() == 'deeplabv3plusatrousresnet152-16os':
        cfg.MODEL_BACKBONE = 'res152_atrous'
        return deeplabv3plus(cfg)
    else:
        raise Exception(f'*** model name wrong, {name} not legal')






if __name__ == '__main__':
    import torch

    model = get_deeplabv3plus_model(name='deeplabv3plusxceptiondensity', n_class=20)
    h, w = 160, 160
    x = torch.rand(2, 3, h, w)
    y = model(x)
    density_map = y[0, 2 + 2:4 + 2]  # 2 x h x w
    density_map[0] = density_map[0].softmax(dim=-1) * float(w) / 1024.
    density_map[1] = density_map[1].softmax(dim=-2) * float(h) / 1024.
    xym_s = density_map  #
    xym_s[0] = xym_s[0].cumsum(dim=-1)  # w->x
    xym_s[1] = xym_s[1].cumsum(dim=-2)  # h->y
    print(y.shape)
    # print(y[0,5].softmax(dim=0))
    print(xym_s)

    xm = torch.linspace(0, 2, 2048).view(
        1, 1, -1).expand(1, 1024, 2048)
    ym = torch.linspace(0, 1, 1024).view(
        1, -1, 1).expand(1, 1024, 2048)
    xym = torch.cat((xm, ym), 0)
    print(xym[:, 1:h, 1:w])
    print(xym[:, 1:h + 1, 1:w + 1] - xym_s)
    pass
