r'''modified source code from DHPF https://github.com/juhongm999/dhpf'''

import torch
import pdb

class KeypointToFlow:
    def __init__(self, receptive_field_size=35, jsz=16, feat_size=16, img_size=256):
        self.feat_size = feat_size
        self.img_size = img_size
        self.box, self.feat_ids = self.receptive_fields(receptive_field_size, jsz, feat_size) # box (N, 2) feat_ids (N, 2)

    def receptive_fields(self, receptive_field_size, jsz, feat_size):
        r"""Returns a set of receptive fields (N, 4)"""
        width = feat_size
        height = feat_size

        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2) # (width*height, 2), (256, 2)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1)

        box = torch.zeros(feat_ids.size()[0], 4) #(x1, y1, x2, y2) (256,4)
        box[:, 0] = feat_ids[:, 1] * jsz - receptive_field_size // 2 + jsz // 2
        box[:, 1] = feat_ids[:, 0] * jsz - receptive_field_size // 2 + jsz // 2
        box[:, 2] = feat_ids[:, 1] * jsz + receptive_field_size // 2 + jsz // 2
        box[:, 3] = feat_ids[:, 0] * jsz + receptive_field_size // 2 + jsz // 2

        return box, feat_ids

    def neighbours(self, box, kps):
        r"""Returns boxes in one-hot format that covers given keypoints"""
        # kps (2, np), box (N, 2)
        box_duplicate = box.unsqueeze(2).repeat(1, 1, len(kps.t())).transpose(0, 1)  # (4, N, np)
        kps_duplicate = kps.unsqueeze(1).repeat(1, len(box), 1) # (2, N, np)

        xmin = kps_duplicate[0].ge(box_duplicate[0]) #(N, np)
        ymin = kps_duplicate[1].ge(box_duplicate[1])
        xmax = kps_duplicate[0].le(box_duplicate[2])
        ymax = kps_duplicate[1].le(box_duplicate[3])

        nbr_onehot = torch.mul(torch.mul(xmin, ymin), torch.mul(xmax, ymax)).t()  # (np, N)
        n_neighbours = nbr_onehot.sum(dim=1) # (np, )
        n_points = nbr_onehot.sum(dim=0) # (N, ) for each box, how many kps are covered inside the box

        return nbr_onehot, n_neighbours, n_points

    def __call__(self, batch):
        src_kps, trg_kps, n_pts = batch['src_kps'].t(), batch['trg_kps'].t(), batch['n_pts'] #src_kps (40,2)

        kp = trg_kps.narrow_copy(0, 0, n_pts) # kp (npts, 2)
        kp_src = src_kps.narrow_copy(0, 0, n_pts) # kp_src (npts, 2)

        src_nbr_onehot, n_neighbours, n_points = self.neighbours(self.box, kp.t()) #self.box (256,4), src_nbr_onehot (np, N), n_points (N, ) 

        center = torch.stack(((self.box[:, 0] + self.box[:, 2])/2, (self.box[:, 1] + self.box[:, 3])/2), dim=1) # (N=256, 2)
        center = center.unsqueeze(0).repeat(len(kp), 1, 1) # (np, N, 2)

        src_idx = src_nbr_onehot.nonzero()  # (M, 2) return id1 (for np), id2 (for N) that is non-zero

        src_nn = center[src_idx[:,0],src_idx[:,1]]
        kp_selected = kp[src_idx[:,0],:] # (M, 2)

        vector_summator = torch.zeros_like(center) # (np, N, 2)
        vector_summator[src_idx[:, 0], src_idx[:, 1]] = kp_selected

        n_points_expanded = n_points.unsqueeze(1).repeat(1,2).float() # (N, 2)
        n_points_expanded[n_points_expanded == 0] = 1.

        source_averaged = (vector_summator.sum(dim=0) / n_points_expanded)[src_idx[:,1]] # (N, 2) / (N, 2) [] -> (M, 2)

        flow = kp_src[src_idx[:,0],:] - source_averaged # (M, 2)

        flow_index = self.feat_ids.index_select(dim=0, index=src_idx[:,1]) # (N, 2) -> (M, 2)

        flow_map = torch.zeros(self.feat_size, self.feat_size, 2)
        flow_map[flow_index[:,0],flow_index[:,1]] = flow / (self.img_size // self.feat_size)

        flow_map = flow_map.permute(2, 0, 1) # (2, feat_size, feat_size) flow_map not normalized, but rescaled to fea size
        
        return flow_map
            