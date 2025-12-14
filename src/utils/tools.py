import torch
import os
import io
import time

def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name='', tag=""):
    dir = os.path.join('pre_trained_models', name)
    os.makedirs(dir, exist_ok=True)
    
    # 将小数点替换为下划线或其他字符
    tag_str = str(tag).replace('.', '_')+'_'+time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    file_path = os.path.join(dir, f'{tag_str}.pt') # 使用 os.path.join 拼接完整文件路径
    torch.save(model.state_dict(), file_path)


def load_model(args, name=''):
    # name = save_load_name(args, name)
    #name = 'best_model'
    with open(f'pre_trained_models/{name}.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model = torch.load(buffer)
    return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)
    
    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)
    
    return t
