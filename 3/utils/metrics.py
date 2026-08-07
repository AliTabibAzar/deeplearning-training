def abs_error_sum(pred, target):
    return (pred - target).abs().sum().item()