import torch


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=2, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.seg = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_planes,
                out_channels=_hidden,
                kernel_size=1,
                stride=1,
            ),
            torch.nn.BatchNorm2d(_hidden),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(
                in_channels=_hidden,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            torch.nn.Sigmoid(),
        )
        self.apply(init_weight)

    def forward(self, x):
        x = self.seg(x)
        return x


class ProjectionMap(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super(ProjectionMap, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=1,
                stride=1,
                bias=True,
            )
        )
        self.apply(init_weight)

    def forward(self, x):
        x = self.layers(x)
        return x


class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        x = self.layers(x)
        return x


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        x = x[:, :, 0]
        x = torch.max(x, dim=1).values
        return x
