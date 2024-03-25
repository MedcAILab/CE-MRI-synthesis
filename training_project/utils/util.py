import torch.nn as nn
from monai.transforms import SobelGradients, Compose


def get_duration_time_str(s_time, e_time):
    h, remainder = divmod((e_time - s_time), 3600)  # 小时和余数
    m, s = divmod(remainder, 60)  # 分钟和秒
    time_str = "%02d h:%02d m:%02d s" % (h, m, s)
    return time_str


def get_edge(tensor):
    # sobel算子和0.4卡阈值
    # tensor.squeeze(0)
    transforms_x = Compose([
        SobelGradients(kernel_size=3, spatial_axes=[-1],
                       padding_mode="zeros"
                       ),
        # AsDiscrete(threshold=0.4)
    ]
    )
    transforms_y = Compose([
        SobelGradients(kernel_size=3, spatial_axes=[-2],
                       padding_mode="zeros"
                       ),
        # AsDiscrete(threshold=0.4)
    ]
    )
    edge_map_x = transforms_x(tensor)
    edge_map_y = transforms_y(tensor)
    # edge_map = torch.pow((torch.pow(edge_map_x, 2) + torch.pow(edge_map_y, 2)), 0.5)
    edge_map = edge_map_y + edge_map_x
    # edge_map.unsqueeze(1)
    return edge_map


def init_weights(net, init_type='kaiming', gain=0.02):
    """
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    """

    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('InstanceNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    # propagate to children
    for m in net.children():
        if hasattr(m, 'init_weights'):
            m.init_weights(init_type, gain)
    return net
