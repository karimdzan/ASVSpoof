from src.metric.base_metric import BaseMetric
from src.metric.utils import compute_eer


class tDCF_EER(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
    def __call__(self, logits, target, **kwargs):
        logits = logits.detach().cpu().numpy()
        logits = logits[..., 1]
        target = target.detach().cpu().numpy()
        eer, _ = compute_eer(logits[target == 1], logits[target == 0])
        return eer