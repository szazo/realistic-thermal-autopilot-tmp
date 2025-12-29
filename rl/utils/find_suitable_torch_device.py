import torch
import logging


def find_suitable_torch_device(device_str: str) -> torch.device:

    log = logging.getLogger('find_suitable_torch_device')
    log.debug('find_suitable_torch_device; device_str=%s', device_str)

    if device_str == 'cpu':
        log.debug('using cpu')
        return torch.device(device_str)

    if device_str == 'max_memory_cuda':

        if not torch.cuda.is_available():
            log.debug('cuda is not available, using cpu')
            return torch.device('cpu')
        
        # find suitable gpu
        log.debug('finding suitable gpu based on memory')
        max_memory = 0
        best_cuda_index = -1
        best_name = None
        for i in range(torch.cuda.device_count()):
            # wait for it to finish its job if any
            torch.cuda.synchronize(i)

            # query memory
            properties = torch.cuda.get_device_properties(i)
            total_memory = properties.total_memory
            if total_memory > max_memory:
                max_memory = total_memory
                best_cuda_index = i
                best_name = properties.name

        if best_cuda_index == -1:
            log.debug('no cuda device found, using cpu')
            return torch.device('cpu')

        device = torch.device(f'cuda:{best_cuda_index}')
        log.debug('found best cuda device "%s" (%s) with memory %.0fGB',
                  best_name, device, max_memory / 1024 / 1024 / 1024)

        return device


    device = torch.device(device_str)
    log.debug('checking device availability %s...', device)

    if device.type == 'cuda':
        if not torch.cuda.is_available():
            log.debug('cuda is not available, using cpu')
            return torch.device('cpu')

        if device.index is not None:
            if device.index >= torch.cuda.device_count():
                log.debug('cuda device with the requested index %s is not available, using cpu; device_count=%s', device.index, torch.cuda.device_count())
                return torch.device('cpu')

    log.debug('using device %s', device)
    return device
