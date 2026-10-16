from datasets.CityscapesDataset import get_cityscapes_dataloader, get_test_dataloader
from datasets.voc import  get_voc_instance_dataloader
from datasets.sbd import  get_sbd_instance_dataloader
from datasets.iSAIDDataset import get_iSAID_dataloader


if __name__ =='__main__':
    from train_config import config
    from torchvision import transforms
    import numpy as np

    args = config()
    data_loader1 = get_voc_instance_dataloader(args)
    print(len(data_loader1))
    data_loader2 = get_voc_instance_dataloader(args, train=False)
    print(len(data_loader2))
    data_loader3 = get_sbd_instance_dataloader(args)
    print(len(data_loader3))