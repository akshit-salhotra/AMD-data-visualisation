batchnorm_stats = {
    'train': {},
    'val': {}
}

def create_hook(name):
    def hook_fn(module, input, output):
        input = input[0]
        dims = [0] + list(range(2, input.dim()))  # dims to reduce over batch and spatial
        batch_mean = input.mean(dim=dims).detach().cpu()
        batch_std = input.std(dim=dims).detach().cpu()
        running_mean = module.running_mean.detach().cpu()
        running_std = module.running_var.sqrt().detach().cpu()

        # Detect if we're in train or val mode
        mode = 'train' if module.training else 'val'

        if name not in batchnorm_stats[mode]:
            batchnorm_stats[mode][name] = {
                "batch_mean": [],
                "batch_std": [],
                "running_mean": [],
                "running_std": []
            }
        
        batchnorm_stats[mode][name]["batch_mean"].append(batch_mean)
        batchnorm_stats[mode][name]["batch_std"].append(batch_std)
        batchnorm_stats[mode][name]["running_mean"].append(running_mean)
        batchnorm_stats[mode][name]["running_std"].append(running_std)
    return hook_fn