import torch
import torch.nn.functional as F


def entropy(output):
    batch_size = output.shape[0]
    entropy = -torch.sum(torch.log(output + 1e-8) * output) / batch_size
    return entropy.item()


def ece(output, target):
    _ece = 0.0

    output = F.softmax(output, dim=-1)
    confidences, predictions = torch.max(output, 1)
    accuracies = predictions.eq(target)

    bin_boundaries = torch.linspace(0, 1, 10 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * \
                 confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            _ece += torch.abs(avg_confidence_in_bin -
                              accuracy_in_bin) * prop_in_bin
    _ece = _ece if isinstance(_ece, float) else _ece.item()
    return _ece


def evaluate(output, target):
    with torch.no_grad():
        _loss = F.nll_loss(output, target, reduction='sum').item()
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        _ece = ece(output, target)
        _entropy = entropy(output)
        _error = error(pred, target)
        return _error, _ece, _entropy, _loss


def error(output, target):
    batch_size = output.shape[1]
    correct = output.eq(target.view(1, -1).expand_as(output))
    correct_k = correct[:1].view(-1).float().sum(0)
    res = 100 - correct_k.mul_(100.0 / batch_size)
    return res.float().item()



