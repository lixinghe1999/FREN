import torch
def show(state):
    for k, v in state.items():
        print(k)
        print(v.shape)
checkpoint = torch.load("logs/blender_chair_hashXYZ_sphereVIEW_fine512_log2T19_lr0.01_decay10_RAdam_sparse1e-10_TV1e-06/010000.tar")
step = checkpoint['global_step']
fn = checkpoint['network_fn_state_dict']
show(fn)
fine = checkpoint['network_fine_state_dict']
show(fine)
embed_fn = checkpoint['embed_fn_state_dict']
show(embed_fn)
opt = checkpoint['optimizer_state_dict']
torch.save({
    'network_fn_state_dict': fn,
    'network_fine_state_dict': fine,
    'embed_fn_state_dict': embed_fn,
}, 'clear.tar')
