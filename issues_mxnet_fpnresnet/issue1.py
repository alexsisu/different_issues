import mxnet as mx
from gluoncv import model_zoo, data

net = model_zoo.get_model('faster_rcnn_fpn_resnet101_v1d_coco', pretrained=True)



all_images = [f"frame_{p}.jpg" for p in range(1,6)]

x, orig_img = data.transforms.presets.rcnn.load_test(all_images)

xx = mx.nd.stack(*[p[0] for p in  x])
res= net(xx)

