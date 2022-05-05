import numpy as np
import torch
from nets.yolo import YoloBody


def transform_to_darknet(pth_file, weights_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(3, num_classes=n_classes)

    pretrained_dict = torch.load(pth_file, map_location=torch.device(device))
    model.load_state_dict(pretrained_dict)
    model.eval()

    fp = open(weights_file, "wb")

    # write head infomation into the file
    header_info = np.array([0, 2, 0, 32013312, 0], dtype=np.int32)
    header_info.tofile(fp)

    # iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def["type"] == "convolutional":
            conv_layer = module[0]

            # if batch norm, load bn first
            if module_def["batch_normalize"]:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

if __name__ == '__main__':
    pth_file = 'model_data/ep100-loss1.539-val_loss1.490.pth' #sys.argv[1]
    weights_file = 'model_data/custom-yolov4-tiny-detector.weights'
    batch_size = 1 #int(sys.argv[3])
    n_classes = 2 #int(sys.argv[4])
    IN_IMAGE_H = 416 #int(sys.argv[5])
    IN_IMAGE_W = 416 #int(sys.argv[6])

    transform_to_darknet(pth_file, weights_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
