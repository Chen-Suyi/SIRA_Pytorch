import torch


def tensor_gpu(batch, local_rank, check_on=True):
    def check_on_gpu_list(list_tensor_: list):
        list_tensor_g = []
        for tensor_ in list_tensor_:
            if isinstance(tensor_, torch.Tensor):
                list_tensor_g.append(
                    tensor_.cuda(local_rank, non_blocking=True))
            elif isinstance(tensor_, list):
                list_tensor_g.append(check_on_gpu_list(tensor_))
            elif isinstance(tensor_, dict):
                list_tensor_g.append(check_on_gpu_dict(tensor_))
            else:
                list_tensor_g.append(tensor_)
        return list_tensor_g

    def check_on_gpu_dict(dict_tensor_: dict):
        dict_tensor_g = {}
        for key, tensor_ in dict_tensor_.items():
            if isinstance(tensor_, torch.Tensor):
                dict_tensor_g[key] = tensor_.cuda(local_rank,
                                                  non_blocking=True)
            elif isinstance(tensor_, list):
                dict_tensor_g[key] = check_on_gpu_list(tensor_)
            elif isinstance(tensor_, dict):
                dict_tensor_g[key] = check_on_gpu_dict(tensor_)
            else:
                dict_tensor_g[key] = tensor_
        return dict_tensor_g

    def check_on_gpu(tensor_):
        if isinstance(tensor_, torch.Tensor):
            tensor_g = tensor_.cuda(local_rank, non_blocking=True).float()
        elif isinstance(tensor_, list):
            tensor_g = check_on_gpu_list(tensor_)
        elif isinstance(tensor_, dict):
            tensor_g = check_on_gpu_dict(tensor_)
        else:
            tensor_g = tensor_
        return tensor_g

    def check_off_gpu_list(list_tensor_: list):
        list_tensor_c = []
        for tensor_ in list_tensor_:
            if isinstance(tensor_, torch.Tensor):
                list_tensor_c.append(tensor_.cpu())
            elif isinstance(tensor_, list):
                list_tensor_c.append(check_off_gpu_list(tensor_))
            elif isinstance(tensor_, dict):
                list_tensor_c.append(check_off_gpu_dict(tensor_))
            else:
                list_tensor_c.append(tensor_)
        return list_tensor_c

    def check_off_gpu_dict(dict_tensor_: dict):
        dict_tensor_g = {}
        for key, tensor_ in dict_tensor_.items():
            if isinstance(tensor_, torch.Tensor):
                dict_tensor_g[key] = tensor_.cpu()
            elif isinstance(tensor_, list):
                dict_tensor_g[key] = check_off_gpu_list(tensor_)
            elif isinstance(tensor_, dict):
                dict_tensor_g[key] = check_off_gpu_dict(tensor_)
            else:
                dict_tensor_g[key].append(tensor_)
        return dict_tensor_g

    def check_off_gpu(tensor_):
        if isinstance(tensor_, torch.Tensor):
            if tensor_.is_cuda:
                tensor_c = tensor_.cpu()
            else:
                tensor_c = tensor_
        elif isinstance(tensor_, list):
            tensor_c = check_off_gpu_list(tensor_)
        elif isinstance(tensor_, dict):
            tensor_c = check_off_gpu_dict(tensor_)
        else:
            tensor_c = tensor_
        return tensor_c

    if torch.cuda.is_available():
        if check_on:
            batch = check_on_gpu(batch)
        else:
            batch = check_off_gpu(batch)
    else:
        batch = batch

    return batch