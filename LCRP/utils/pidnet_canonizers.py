from copy import deepcopy
from LCRP.utils.base_canonizers import SequentialThreshCanonizer
import zennit.canonizers as zcanon
from zennit.layer import Sum
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
from zennit.composites import EpsilonPlusFlat
from zennit.layer import Sum
from zennit.rules import Epsilon



algc = False

class InterpolateWrapper(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x, 
            size=self.size, 
            scale_factor=self.scale_factor, 
            mode=self.mode, 
            align_corners=self.align_corners
        )

# Canonizer for PIDNet
class PIDNetBaseCanonizer(zcanon.AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    def apply(self, root_module):
        '''Overload the attributes for all applicable modules.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            Root module for which underlying modules will have their attributes overloaded.

        Returns
        -------
        instances : list of obj:`Canonizer`
            The applied canonizer instances, which may be removed by calling `.remove`.
        '''
        instances = []
        for name, module in root_module.named_modules():
            attributes = self.attribute_map(name, module)
            if attributes is not None:
                instance = self.__class__() # this should be changed in zennit :/
                instance.register(module, attributes)
                instances.append(instance)
        return instances
    @classmethod
    def _attribute_map(cls, name, module):
        # BasicBlock
        if module.__class__.__name__ == "BasicBlock":
            return {
                'forward': cls.forward_basicblock.__get__(module),
                'canonizer_sum': Sum()
            }

        # Bottleneck
        if module.__class__.__name__ == "Bottleneck":
            return {
                'forward': cls.forward_bottleneck.__get__(module),
                'canonizer_sum': Sum()
            }

        # PAPPM
        if module.__class__.__name__ == "PAPPM":
            return {
                'forward': cls.forward_pappm.__get__(module),
                'canonizer_sum': Sum(),
                'orig_scale_process_params': cls.get_conv_layer_params(module.scale_process[2]),
                'scale_process': cls.convert_grouped_conv_to_regular(module.scale_process)
            }

        # Light_Bag
        if module.__class__.__name__ == "Light_Bag":
            return {
                'forward': cls.forward_lightbag_branch_i_only.__get__(module),
                'canonizer_sum': Sum()
            }

        return None
    @staticmethod
    def get_conv_layer_params(conv_g):
        return {
            "init": {"in_channels": conv_g.in_channels,
            "out_channels": conv_g.out_channels,
            "kernel_size": conv_g.kernel_size,
            "stride": conv_g.stride,
            "padding": conv_g.padding,
            "dilation": conv_g.dilation,
            "bias": (conv_g.bias is not None),
            "groups": conv_g.groups},
            "params":{
                "weight": conv_g.weight.data.detach(),
                "bias": conv_g.bias.data.detach() if conv_g.bias is not None else None
            }
        }
    def remove(self):
        if "orig_scale_process_params" in self.attribute_keys:
            mdl = nn.Conv2d(**self.module.orig_scale_process_params["init"])
            mdl.weight.data = self.module.orig_scale_process_params["params"]["weight"]
            if self.module.orig_scale_process_params["params"]["bias"] is not None:
                mdl.bias.data = self.module.orig_scale_process_params["params"]["bias"]
            self.module.scale_process[2] = mdl
        for key in self.attribute_keys:
            if key !="scale_process":
                delattr(self.module, key)

    @staticmethod
    def convert_grouped_conv_to_regular(seq):
        new_seq=deepcopy(seq)
        conv_g=seq[2]
        G = conv_g.groups
        Cin_per_group = conv_g.in_channels // G
        Cout_per_group = conv_g.out_channels // G

        # Create equivalent regular conv
        conv_regular = nn.Conv2d(
            in_channels=conv_g.in_channels,
            out_channels=conv_g.out_channels,
            kernel_size=conv_g.kernel_size,
            stride=conv_g.stride,
            padding=conv_g.padding,
            dilation=conv_g.dilation,
            bias=(conv_g.bias is not None),
            groups=1
        )

        # Zero all weights first
        with torch.no_grad():
            conv_regular.weight.zero_()

            # Copy group weights into corresponding block
            for g in range(G):
                out_start = g * Cout_per_group
                in_start = g * Cin_per_group

                conv_regular.weight[
                    out_start : out_start + Cout_per_group,
                    in_start : in_start + Cin_per_group,
                ] = conv_g.weight[out_start : out_start + Cout_per_group]

            # Copy biases
            if conv_g.bias is not None:
                conv_regular.bias.copy_(conv_g.bias)
        new_seq[2] = conv_regular
        return new_seq
    
    @staticmethod
    def forward_basicblock(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.canonizer_sum(torch.stack([out, residual], dim=-1))

        if self.no_relu:
            return out
        else:
            return self.relu(out)


    @staticmethod
    def forward_bottleneck(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.canonizer_sum(torch.stack([out, residual], dim=-1))
        if self.no_relu:
            return out
        else:
            return self.relu(out)

    @staticmethod
    def forward_pappm(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []
        self.interp1 = InterpolateWrapper(size=[height, width], mode='bilinear', align_corners=algc)
        self.interp2 = InterpolateWrapper(size=[height, width],mode='bilinear', align_corners=algc)
        self.interp3 = InterpolateWrapper(size=[height, width],mode='bilinear', align_corners=algc)
        self.interp4 = InterpolateWrapper(size=[height, width],mode='bilinear', align_corners=algc)

        x_ = self.scale0(x)

        s1 = self.interp1(self.scale1(x))
        s2 = self.interp2(self.scale2(x))
        s3 = self.interp3(self.scale3(x))
        s4 = self.interp4(self.scale4(x))
        
        scale_list.append(self.canonizer_sum(torch.stack([s1, x_], dim=-1)))
        scale_list.append(self.canonizer_sum(torch.stack([s2, x_], dim=-1)))
        scale_list.append(self.canonizer_sum(torch.stack([s3, x_], dim=-1)))
        scale_list.append(self.canonizer_sum(torch.stack([s4, x_], dim=-1)))
        # scale_list.append(self.canonizer_sum(torch.stack([s2, x_], dim=-1)))

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        # Here is some error with gradient, that dimensions do not correspond (on forward pass no problem).
        out = self.compression(torch.cat([x_,scale_out],1)) + self.shortcut(x)
        return out
    
    @staticmethod
    def forward_lightbag_branch_i_only(self, p, i, d):
        # Detaching branches P and D here 
        edge_att = torch.sigmoid(d).detach()
        
        p_add = self.conv_p((1-edge_att)*i + p.detach())
        i_add = self.conv_i(i + edge_att*p.detach())
        
        return self.canonizer_sum(torch.stack([p_add, i_add], dim=-1))


# Top-level composite canonizer to combine canonization strategies
class PIDNetCanonizer(zcanon.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            PIDNetBaseCanonizer(),
            SequentialThreshCanonizer(),
        ))

class EpsilonPlusFlatforPIDNet(EpsilonPlusFlat):
    def __init__(self, canonizers=None):
        super().__init__(canonizers=canonizers)
        self.layer_map.append((InterpolateWrapper, Epsilon()))