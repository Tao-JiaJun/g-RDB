from model.backbone.resnet import resnet18, resnet50
from model.backbone.repvgg import RepVGG_A0, RepVGG_B0
from model.neck.FPN import FPN
from model.neck.RDBFPN import RDBFPN

def get_backbone(name, pretrained):
    if name == 'RESNET18':
        if pretrained:
            return resnet18(pretrained=True)
        else:
            return resnet18()
    elif name == 'RESNET50':
        if pretrained:
            return resnet50(pretrained=True)
        else:
            return resnet50()
    elif name == "A0":
        return RepVGG_A0(pretrained=True,freeze_bn=False)
    if name == "B0":
        return RepVGG_B0(pretrained=True,freeze_bn=False)
    else:
        raise Exception('Backbone name error!')
def get_fpn(param, boost):
    name = param["NAME"]
    if name == 'FPN':
        return FPN(C3_size=param["C3_CHANNEL"], 
                   C4_size=param["C4_CHANNEL"], 
                   C5_size=param["C5_CHANNEL"], 
                   feature_size=param["OUT_CHANNEL"])
    elif name == 'RDB':
        return RDBFPN(inplanes=[param["C3_CHANNEL"],param["C4_CHANNEL"]],
                      feature_size=param["OUT_CHANNEL"],
                      depth=param["DEPTH"],
                      groups=param["GROUPS"],
                      use_at=param["USE_AT"],
                      boost=boost)
    else:
        raise Exception('Neck name error!')