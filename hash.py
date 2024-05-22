import torch
import torch
import torch.nn as nn

def trilinear_Interpolate(x, cube_size, bottle_left_vertex, voxel_hash_value):
     # voxel_hash_value : [B, 8, 2]
    top_rigtht_vertex = bottle_left_vertex + torch.tensor([1.0, 1.0, 1.0]).to('cuda') * cube_size
    # print(x.shape," ",bottle_left_vertex.shape," ",voxel_hash_value.shape)
    # torch.Size([32000, 5, 2])   torch.Size([32000, 5, 2])   torch.Size([32000, 4, 5, 1])
    # weights.shape: torch.Size([32000, 5, 2])
    # y0.shape:  torch.Size([32000, 5, 2])

    # [B, 3]
    weights = (x - bottle_left_vertex) / (top_rigtht_vertex - bottle_left_vertex);
    # print("weights.shape:",weights.shape)
    # print("voxel_hash_value[:,0]: ",voxel_hash_value[:,0].shape, " weights[:,:,0][:,:, None]: ",weights[:,:,0][:, None].shape)
    x0 = voxel_hash_value[:,0] * (1 - weights[:,:,0])[:,:, None] + voxel_hash_value[:, 4] * weights[:,:,0][:,:, None];
    x1 = voxel_hash_value[:,1] * (1 - weights[:,:,0])[:,:, None] + voxel_hash_value[:, 5] * weights[:,:,0][:,:, None];
    x2 = voxel_hash_value[:,2] * (1 - weights[:,:,0])[:,:, None] + voxel_hash_value[:, 6] * weights[:,:,0][:,:, None];
    x3 = voxel_hash_value[:,3] * (1 - weights[:,:,0])[:,:, None] + voxel_hash_value[:, 7] * weights[:,:,0][:,:, None];
    
    y0 = x0 * (1 - weights[:,:,1])[:,:, None] + x2 * weights[:,:,1][:,:, None]
    y1 = x1 * (1 - weights[:,:,1])[:,:, None] + x3 * weights[:,:, 1][:,:, None]
    # print("y0.shape: ",y0.shape)
    z = y0 * (1 - weights[:,:,2])[:,:,None] + y1 * weights[:,:,2][:,:, None]
    return z




@torch.no_grad()
def spatial_hash(x, T):
    primes = [1, 2654435761, 805459861]
    result = torch.zeros_like(x)[..., 0]
    for i in range(x.shape[-1]):
        result ^= x[..., i] * primes[i]
    return result % T
    


def hashing_of_voxel(max_bound, min_bound, x, N_l, T):

    cube_size = (max_bound - min_bound)/N_l;

    bottom_left_index = torch.floor((x - min_bound)/cube_size).int()
    bottom_left_vertex = bottom_left_index * cube_size + min_bound;
    voxel_hash_indices = [] 

    vertex_0 = bottom_left_index + torch.tensor([0, 0, 0]).to('cuda')
    vertex_1 = bottom_left_index + torch.tensor([0, 0, 1]).to('cuda')
    vertex_2 = bottom_left_index + torch.tensor([0, 1, 0]).to('cuda')
    vertex_3 = bottom_left_index + torch.tensor([0, 1, 1]).to('cuda')
    vertex_4 = bottom_left_index + torch.tensor([1, 0, 0]).to('cuda')
    vertex_5 = bottom_left_index + torch.tensor([1, 0, 1]).to('cuda')
    vertex_6 = bottom_left_index + torch.tensor([1, 1, 0]).to('cuda')
    vertex_7 = bottom_left_index + torch.tensor([1, 1, 1]).to('cuda')

    voxel_hash_indices.append(spatial_hash(vertex_0, T))
    voxel_hash_indices.append(spatial_hash(vertex_1, T))
    voxel_hash_indices.append(spatial_hash(vertex_2, T))
    voxel_hash_indices.append(spatial_hash(vertex_3, T))
    voxel_hash_indices.append(spatial_hash(vertex_4, T))
    voxel_hash_indices.append(spatial_hash(vertex_5, T))
    voxel_hash_indices.append(spatial_hash(vertex_6, T))
    voxel_hash_indices.append(spatial_hash(vertex_7, T))
    return cube_size, bottom_left_vertex, torch.stack(voxel_hash_indices).transpose(0,1)

class HashEncoder(nn.Module):
    def __init__(self, bounding_box):
        super(HashEncoder, self).__init__()
        self.min_bound, self.max_bound = bounding_box
        self.logT = 14;
        self.L = 16;
        self.T = 2 ** 14;
        self.F = 1;
        self.N_min =  torch.tensor(16);
        self.N_max =  torch.tensor(512);

        self.b = torch.exp((torch.log(self.N_max)-torch.log(self.N_min))/(self.L-1))
        hash_tables = [];
        for i in range(self.L):
            # [T, F]
            hash_table = nn.Embedding(self.T, self.F);
            nn.init.uniform_(hash_table.weight, a=-0.0001, b=0.0001)
            hash_tables.append(hash_table)

        self.hash_tables = nn.ModuleList(hash_tables);


    def forward(self, x):
        feature_vector = [];
        for i in range(self.L):
            N_l = torch.floor(self.N_min * self.b**i)
            cube_size, bottle_left_vertex, voxel_hash_indices = hashing_of_voxel(self.max_bound, self.min_bound, x, N_l, self.T)
            voxel_hash_value = self.hash_tables[i](voxel_hash_indices)
            feature_vector.append(trilinear_Interpolate(x, cube_size, bottle_left_vertex, voxel_hash_value))

        return torch.cat(feature_vector, dim=-1)

