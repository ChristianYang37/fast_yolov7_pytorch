import time
import torch
import torch_pruning as tp


def prune_model(model, device, opt):
    model.eval()
    now = time.time()
    for _ in range(10):
        model(torch.randn(32, 3, 224, 224).to(device))
    before_runtime = (time.time() - now) / 10

    print(model)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    imp = tp.importance.MagnitudeImportance(p=2 if opt.prune_norm == 'L2' else 1)  # L2 norm pruning

    ignored_layers = []
    from models.yolo import Detect, IDetect
    from models.common import ImplicitA, ImplicitM
    for m in model.modules():
        if isinstance(m, (Detect, IDetect)):
            ignored_layers.append(m.m)
    unwrapped_parameters = []
    for m in model.modules():
        if isinstance(m, (ImplicitA, ImplicitM)):
            unwrapped_parameters.append((m.implicit, 1))  # pruning 1st dimension of implicit matrix

    iterative_steps = 1  # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=opt.sparsity,
        # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
        # unwrapped_parameters=unwrapped_parameters
    )
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    pruner.step()

    now = time.time()
    for _ in range(10):
        model(torch.randn(32, 3, 224, 224).to(device))
    after_runtime = (time.time() - now) / 10

    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print("Pruning Sparsity=sparsity%f" % opt.sparsity)
    print("Before Pruning: MACs=%f G, #Params=%f G, batch 32 average time=%f" % (
    base_macs / 1e9, base_nparams / 1e9, before_runtime))
    print("After Pruning: MACs=%f G, #Params=%f G, batch 32 average time=%f" % (
    pruned_macs / 1e9, pruned_nparams / 1e9, after_runtime))
