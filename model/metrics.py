import statistics
import typing

import torch


class ImitationLearningPointwiseMetrics:
    def __init__(self,
                 affordances: torch.Tensor,
                 grasping_index: typing.Dict,
                 successful: torch.Tensor,
                 losses: typing.Optional[torch.Tensor] = None):
        self.affordances = affordances.clone().detach().cpu()
        self.losses = losses.clone().detach().cpu() if losses is not None else None
        self.grasping_index = grasping_index
        self.given_logits = torch.tensor([affordances[i, 0, r, c, a] for i, (r, c, a) in enumerate(zip(grasping_index["row"], grasping_index["col"], grasping_index["angle_index"]))])
        self.given_labels = successful.clone().detach().cpu()
        self.distances = self.calculate_distances()

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
        affordances = torch.stack([self.affordances[i, ..., angle_idx] for i, angle_idx in enumerate(self.grasping_index["angle_index"])])
        affordances = affordances[:, 0] * (self.distances < max_distance)
        affordances /= torch.sum(affordances, dim=(1, 2)).reshape(affordances.shape[0], 1, 1)

        return torch.sum(self.distances * affordances) / self.affordances.shape[0]

    def entropy(self):
        return -torch.sum(self.affordances.reshape(-1) * torch.log(self.affordances.reshape(-1)))

    def ranking(self):
        return statistics.mean([
            torch.sum(self.affordances[i] > self.affordances[i, 0, r, c, a]).item()
            for i, (r, c, a) in
            enumerate(zip(self.grasping_index["row"], self.grasping_index["col"], self.grasping_index["angle_index"]))
        ]) / self.affordances.shape[2] / self.affordances.shape[3] / self.affordances.shape[4]

    def summary(self) -> typing.Dict[str, float]:
        return {
            "expected-distance": self.expected_distance(),
            "expected-distance-0.2-max": self.expected_distance(max_distance=0.2),
            "expected-distance-0.1-max": self.expected_distance(max_distance=0.1),
            "entropy": self.entropy(),
            "ranking": self.ranking(),
            "affordance": self.affordance(),
            "loss": torch.mean(self.losses).item(),
        }
