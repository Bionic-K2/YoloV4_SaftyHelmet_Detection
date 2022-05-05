import torch
from nets.yolo import YoloBody


def transform_to_onnx(weight_file, onnx_file_name, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(3, num_classes=n_classes)

    pretrained_dict = torch.load(weight_file, map_location=torch.device(device))
    model.load_state_dict(pretrained_dict)
    model.eval()

    input_names = ["input"]
    output_names = ['boxes', 'confs']

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')

    else:
        x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          opset_version=11,
                          input_names=input_names, 
                          output_names=output_names
                          )

        print('Onnx model exporting done')
    



if __name__ == '__main__':
    weight_file = 'model_data/ep100-loss1.539-val_loss1.490.pth' #sys.argv[1]
    onnx_file = 'model_data/yolov4_test.onnx'
    batch_size = 1 #int(sys.argv[3])
    n_classes = 2 #int(sys.argv[4])
    IN_IMAGE_H = 416 #int(sys.argv[5])
    IN_IMAGE_W = 416 #int(sys.argv[6])

    transform_to_onnx(weight_file, onnx_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
