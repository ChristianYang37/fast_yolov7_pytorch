import time
import torch
import torch_pruning as tp
from models.yolo import Detect, IDetect
from models.common import ImplicitA, ImplicitM


class pruner:
    def __init__(self, model, device, opt):
        model.eval()
        example_inputs = torch.randn(1, 3, 224, 224).to(device)
        imp = tp.importance.MagnitudeImportance(p=2 if opt.prune_norm == 'L2' else 1)  # L2 norm pruning

        ignored_layers = []
        for m in model.modules():
            if isinstance(m, (Detect, IDetect)):
                ignored_layers.append(m.m)
        unwrapped_parameters = []
        for m in model.modules():
            if isinstance(m, (ImplicitA, ImplicitM)):
                unwrapped_parameters.append((m.implicit, 1))  # pruning 1st dimension of implicit matrix

        iterative_steps = opt.epochs // opt.num_epochs_to_prune  # progressive pruning
        self.pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            ch_sparsity=opt.sparsity,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
        )
        self.sparsity = opt.sparsity
        self.num_steps = iterative_steps
        self.count = 0

    def step(self, model, device):
        self.count += 1

        example_inputs = torch.randn(1, 3, 224, 224).to(device)
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

        self.pruner.step()

        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print("Pruning Sparsity=%f" % (self.sparsity / self.num_steps * self.count))
        print("Before Pruning: MACs=%f, #Params=%f" % (base_macs, base_nparams))
        print("After Pruning: MACs=%f, #Params=%f" % (pruned_macs, pruned_nparams))
