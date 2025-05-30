import copy
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, idx_mask=None):
        for t in self.transforms:
            data, idx_mask = t(data, idx_mask)
        return data, idx_mask

_TRANSFORM_DICT = {}

def register_transform(name):
    def decorator(cls):
        _TRANSFORM_DICT[name] = cls
        return cls
    return decorator

def get_transform(cfg):
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = _TRANSFORM_DICT[t_dict.pop('type')]
        tfms.append(cls(**t_dict))  # here with initialize
    return Compose(tfms)

def _index_select(v, index, n):
    if isinstance(v, torch.Tensor) and v.size(0) == n:
        return v[index]
    elif isinstance(v, list) and len(v) == n:
        return [v[i] for i in index]
    else:
        return v


def _index_select_data(data, index):
    return {
        k: _index_select(v, index, data['aa'].size(0))
        for k, v in data.items()
    }


def _mask_select(v, mask):
    if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
        return v[mask]
    elif isinstance(v, list) and len(v) == mask.size(0):
        return [v[i] for i, b in enumerate(mask) if b]
    else:
        return v


def _mask_select_data(data, mask):
    return {
        k: _mask_select(v, mask)
        for k, v in data.items()
    }


# def _get_CB_positions(pos_atoms, mask_atoms):
#     """
#     Args:
#         pos_atoms:  (L, A, 3)
#         mask_atoms: (L, A)
#     """
#     from rde.utils.protein.constants import BBHeavyAtom
#     L = pos_atoms.size(0)
#     pos_CA = pos_atoms[:, BBHeavyAtom.CA]   # (L, 3)
#     if pos_atoms.size(1) < 5:
#         return pos_CA
#     pos_CB = pos_atoms[:, BBHeavyAtom.CB]
#     mask_CB = mask_atoms[:, BBHeavyAtom.CB, None].expand(L, 3)
#     return torch.where(mask_CB, pos_CB, pos_CA)

def _get_CB_positions(pos_atoms, mask_atoms):
    """
    Args:
        pos_atoms:  (L, A, 3)
        mask_atoms: (L, A)
    """
    from DDAffinity.utils.protein.constants import BBHeavyAtom
    B = pos_atoms.size(0)
    L = pos_atoms.size(1)
    pos_CA = pos_atoms[:, BBHeavyAtom.CA]   # (L, 3)

    # 计算虚拟Cb
    # ['N', 'CA', 'C', 'O', 'CB']
    b = pos_atoms[:, 1] - pos_atoms[:, 0]
    c = pos_atoms[:, 2] - pos_atoms[:, 1]
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + pos_atoms[:,  1]  # 虚拟Cb原子

    if pos_atoms.shape[1] < 5:
        return pos_CA
    pos_CB = pos_atoms[:, BBHeavyAtom.CB]
    mask_CB = mask_atoms[:, BBHeavyAtom.CB, None].expand(B, 3)
    # return torch.where(mask_CB, pos_CB, pos_CA)
    return torch.where(mask_CB, pos_CB, Cb)