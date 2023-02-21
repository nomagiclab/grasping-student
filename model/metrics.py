import statistics
import typing
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score

import torch
from torch import conv3d


def topn_masks(tmap, n=5, r=10):
    tmap = tmap.clone().detach()
    if tmap.ndim != 5:
        raise ValueError(f"map should have ndim == 5, shape is: {tmap.shape}")

    device = tmap.device
    bs, _, h, w, a = tmap.shape
    tmap_ = tmap[:, 0]

    # kernel is used to create rectangular shapes of size 2 * r + 1
    # with the middle where values 1 were
    kernel = torch.ones(bs, 1, 2 * r + 1, 2 * r + 1, a, device=device)
    result = torch.full_like(tmap, 0, device=device)
    results = []

    for i in range(n):
        where_max = tmap_.view(bs, -1).max(dim=1)[1]
        max_mask = torch.zeros(bs, h * w * a, device=device)
        max_mask.scatter_(1, where_max[:, None], 1)
        max_mask = max_mask.view(tmap_.shape)

        patches = conv3d(max_mask, kernel, padding='same', groups=bs)

        tmap_[patches == 1] = -1e20
        result.logical_or_((max_mask == 1).reshape(tmap.shape))
        results.append(result == 1)

    if result.sum() != bs * n:
        raise ValueError(f"some fields were selected twice. It's probably\
       because tmap is too large in comparison to n, r. Expected selected: {bs * n}, got {result.sum()}")

    return results

class ImitationLearningPointwiseMetrics:
    def __init__(self,
                 affordances: torch.Tensor,
                 grasping_index: typing.Dict,
                 successful: torch.Tensor,
                 losses: typing.Optional[torch.Tensor] = None,
                 ground_truth: typing.Optional[torch.Tensor] = None):
        self.affordances = affordances#.clone().detach().cpu()
        self.losses = losses#.clone().detach().cpu() if losses is not None else None
        self.grasping_index = grasping_index
        self.given_logits = torch.tensor([affordances[i, 0, r, c, 0] for i, (r, c, a) in enumerate(zip(grasping_index["row"], grasping_index["col"], grasping_index["angle_index"]))])
        self.given_labels = successful#.clone().detach().cpu()
        self.distances = self.calculate_distances().to(affordances.device)
        self.masks = topn_masks(self.affordances)
        self.hits = [mask * ground_truth for mask in self.masks]
        self.ground_truth = ground_truth

    def affordance(self):
        return torch.mean(self.given_logits)

    def calculate_distances(self):
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, steps=self.affordances.shape[2]),
            torch.linspace(0, 1, steps=self.affordances.shape[3]),
            indexing='ij'
        )

        def l2_distance(lhs, rhs):
            return torch.sqrt(torch.sum((lhs - rhs) ** 2, dim=1))

        distances = l2_distance(
            torch.stack([torch.stack([grid_x, grid_y])] * self.affordances.shape[0]),
            torch.stack([
                (self.grasping_index["row"].cpu() / self.affordances.shape[2]),
                (self.grasping_index["col"].cpu() / self.affordances.shape[3]),
            ], dim=1).reshape((self.affordances.shape[0], 2, 1, 1))
        )
        return distances

    def expected_distance(self, max_distance=1.):
        affordances = torch.stack([self.affordances[i, ..., 0] for i, angle_idx in enumerate(self.grasping_index["angle_index"])])
        affordances = affordances[:, 0] * (self.distances < max_distance)
        affordances /= torch.sum(affordances, dim=(1, 2)).reshape(affordances.shape[0], 1, 1)

        return torch.sum(self.distances * affordances) / self.affordances.shape[0]

    def entropy(self):
        return -torch.sum(self.affordances.reshape(-1) * torch.log(self.affordances.reshape(-1)))

    def ranking(self):
        return statistics.mean([
            torch.sum(self.affordances[i] > self.affordances[i, 0, r, c, 0]).item()
            for i, (r, c, a) in
            enumerate(zip(self.grasping_index["row"], self.grasping_index["col"], self.grasping_index["angle_index"]))
        ]) / self.affordances.shape[2] / self.affordances.shape[3] / self.affordances.shape[4]

    def summary(self) -> typing.Dict[str, float]:
        return {
            "random-success-rate": (torch.sum(self.ground_truth) / torch.sum(torch.ones_like(self.ground_truth))).item(),
            "expected-distance": self.expected_distance(),
            "expected-distance-0.2-max": self.expected_distance(max_distance=0.2),
            "expected-distance-0.1-max": self.expected_distance(max_distance=0.1),
            "entropy": self.entropy(),
            "ranking": self.ranking(),
            "affordance": self.affordance(),
            "success-rate": (torch.sum(self.hits[0]) / self.hits[0].shape[0]).item(),
            "top-5-success-rate": (torch.sum(self.hits[-1]) / self.hits[-1].shape[0] / 5).item(),
            "top-5-max-rate": (sum([torch.max(hit).item() for hit in self.hits[-1]]) / self.hits[-1].shape[0]),
            "loss": torch.mean(self.losses).item(),
        }


class AffordanceLearningPointwiseMetrics:
    def __init__(self,
                 affordances: torch.Tensor,
                 grasping_index: typing.Dict,
                 successful: torch.Tensor,
                 losses: typing.Optional[torch.Tensor] = None):
        self.losses = losses.clone().detach().cpu() if losses is not None else None
        self.given_labels = successful.clone().detach().cpu()
        self.given_logits = torch.tensor([affordances[i, 0, r, c, 0] for i, (r, c) in enumerate(zip(grasping_index["row"], grasping_index["col"]))])

    def summary(self) -> typing.Dict[str, float]:
        y_pred = torch.nn.Sigmoid()(self.given_logits) > 0.5
        pr, r, f, s = precision_recall_fscore_support(self.given_labels, y_pred)
        acc = accuracy_score(self.given_labels, y_pred)

        result = {
            "accuracy": acc,
            "loss": torch.mean(self.losses).item(),
        }

        if len(pr) >= 2:
            result["precision"] = pr[1]
            result["recall"] = r[1]
            result["f1-score"] = f[1]
            result["support"] = s[1]
            result["bacc"] = balanced_accuracy_score(self.given_labels, y_pred)

        return result
