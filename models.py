import sys
import torch
import torch.nn.functional as F
from vit_explain.baselines.ViT.ViT_LRP import VisionTransformer
from functools import partial
from vit_explain.modules.layers_ours import RelProp, safe_divide, LayerNorm
import timm


class GlobalPool(RelProp):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs[:, 1:, :].mean(dim=1).unsqueeze(1)


class GlobalPoolLinear(torch.nn.Linear, RelProp):
    def forward(self, input):
        input = input.permute(0, 2, 1)
        input = F.linear(input, self.weight, self.bias)
        input = input.permute(0, 2, 1)
        return input

    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0).permute(0, 2, 1)
        nx = torch.clamp(self.X, max=0).permute(0, 2, 1)
        R = R.permute(0, 2, 1)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R.permute(0, 2, 1)


class VisionTransformerRETFound(VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, norm_layer=None, **kwargs):
        super().__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            embed_dim = kwargs["embed_dim"]
            assert norm_layer is not None
            self.norm = norm_layer(embed_dim)

            # self.pool = GlobalPoolLinear(197, 1, bias=False)
            self.pool = GlobalPool()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = self.pool(x)
            x = self.norm(x)
            x = x.squeeze(1)
        else:
            x = self.norm(x)
            x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
            x = x.squeeze(1)
        x = self.head(x)
        return x


def vit_large_patch16(**kwargs):
    model = VisionTransformerRETFound(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformerRETFound(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_small_patch16(**kwargs):
    model = VisionTransformerRETFound(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        **kwargs,
    )
    return model


def prepare_model(model_name, checkpoint_path, global_pool=True):
    # model = vit_LRP(pretrained=False, num_classes=1)
    model = sys.modules[__name__].__dict__[model_name](
        img_size=224,
        num_classes=1,
        # drop_path_rate=0.2,
        global_pool=global_pool,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if global_pool:
        checkpoint["model"]["norm.weight"] = checkpoint["model"]["fc_norm.weight"]
        checkpoint["model"]["norm.bias"] = checkpoint["model"]["fc_norm.bias"]
        del checkpoint["model"]["fc_norm.weight"]
        del checkpoint["model"]["fc_norm.bias"]
        # checkpoint["model"]["pool.weight"] = torch.cat((torch.zeros((1,1)),
        #                                                 torch.full((1,196), 1/196)), dim=1)

    model.load_state_dict(checkpoint["model"])
    model.to("cuda")
    model.eval()
    return model


def efficientnetv2_m(checkpoint_path, num_classes=1):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
    model = timm.create_model(
        "efficientnetv2_rw_m", pretrained=False, num_classes=num_classes
    )
    model.load_state_dict(checkpoint, strict=False)
    model.to("cuda")
    model.eval()
    return model


def resnet50(checkpoint_path, num_classes=1):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
    model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint, strict=False)
    model.to("cuda")
    model.eval()
    return model
