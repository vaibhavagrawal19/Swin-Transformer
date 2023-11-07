import torch 
import torch.nn as nn
import torch.nn.functional as F

class PatchMerging(nn.Module):
    def __init__(self, in_channels=128, out_channels=256, merge_dim=(2, 2)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_dim = merge_dim
        self.linear = nn.Linear(self.in_channels * merge_dim[0] * merge_dim[1], self.out_channels)

    def forward(self, x):
        N, C, W, H = x.shape
        assert W % self.merge_dim[0] == 0 and H % self.merge_dim[1] == 0
        # TODO remove this condition, handle the case where the patches are not conforming w.r.t. merging
        x = x.permute(0, 2, 3, 1)
        x = [x[:, i::self.merge_dim[0], j::self.merge_dim[1], :] for i in range(self.merge_dim[0]) for j in range(self.merge_dim[1])]
        x = torch.cat(x, dim=-1)
        x = x.view((-1, self.in_channels * self.merge_dim[0] * self.merge_dim[1]))
        x = self.linear(x)
        x = x.view(N, W // self.merge_dim[0], H // self.merge_dim[1], self.out_channels)
        x = x.permute(0, 3, 1, 2)
        return x
    


class PatchEmbedding(nn.Module):
    def __init__(self, image_dim: tuple, patch_dim: tuple, in_channels: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        padding_x = (patch_dim[0] - (image_dim[0] % patch_dim[0])) % patch_dim[0]
        padding_y = (patch_dim[1] - (image_dim[1] % patch_dim[1])) % patch_dim[1]
        self.padding = (padding_x, padding_y)
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_dim, stride=patch_dim, padding=self.padding)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(start_dim=-2)
        x = torch.transpose(x, -2, -1)
        return x



class SwinTransformer(nn.Module):
    def __init__(self, image_dim: tuple, in_channels=3, n_encoders=1, patch_dim=(4, 4), hidden_dim=128, n_heads=8, out_dim=10):
        super().__init__()
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.n_encoders = n_encoders
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.out_dim = out_dim



if __name__ == "__main__":
    p = PatchMerging(2, 4, (2, 2))
    x = torch.rand(1, 2, 4, 4)
    out = p(x)

