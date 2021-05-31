from models.pix2pix_model import Pix2PixModel
from util.util import tensor2im
from data.base_dataset import HAiR_inference_dataLoad
from opt import opt
import numpy as np

class Driver:
    def __init__(self, num_gpu = -1):
        self.opt = opt
        self.opt.gpu_ids = []
        if num_gpu > 0:
            opt.gpu_ids = [i for i in range(num_gpu)]
        
        self.model = Pix2PixModel(opt)
        self.model.eval()

    def process(self, datas):
        data = HAiR_inference_dataLoad(self.opt, datas)

        # forward : generated <- generated_fake_image
        generated = self.model(data, mode='inference')

        fake_image = tensor2im(generated[0])
        if self.opt.add_feat_zeros or self.opt.add_zeros:
            th = self.opt.add_th
            H, W = self.opt.crop_size, self.opt.crop_size
            fake_image_tmp =\
                fake_image[int(th/2):int(th/2)+H, int(th/2):int(th/2)+W, :]
            fake_image = fake_image_tmp

        return np.uint8(fake_image)

# Driver().process(datas={})