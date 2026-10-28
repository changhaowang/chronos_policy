"""A complete CPU training/inference step using pre-encoded image patches."""

import argparse

import torch

from chronos.policy import ChronosPolicy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--method", choices=("chronos", "dt"), default="chronos")
args = parser.parse_args()

torch.manual_seed(0)
torch.set_num_threads(2)
policy = ChronosPolicy(
    positional_encoding="time" if args.method == "chronos" else "index",
    action_dim=10,
    proprio_dims={"robot0_pose": 10},
    cameras=("camera",),
    feature_dim=32,
    dim=64,
    layers=2,
    heads=2,
    patch_grid=2,
    rotary_pairs=8,
)
observations = {
    "camera": torch.randn(2, 2, 4, 32),  # batch, frames, patches, image features
    "robot0_pose": torch.randn(2, 2, 10),
}
capture_times = {
    name: torch.tensor([[-0.08, -0.02], [-0.05, 0.0]], dtype=torch.float64) for name in observations
}
delay = torch.tensor([[0.0], [0.15]], dtype=torch.float64)
action_times = delay + torch.arange(16, dtype=torch.float64)[None] * 0.05

# In a dataset, targets must be demonstrated actions interpolated at action_times.
# Random tensors below only demonstrate the API; this is not a paper experiment.
targets = torch.randn(2, 16, 10).clamp(-1, 1)
optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
optimizer.zero_grad()
times = capture_times if args.method == "chronos" else None
query_times = action_times if args.method == "chronos" else None
loss = policy.loss(observations, times, query_times, targets)
loss.backward()
optimizer.step()

policy.eval()
actions = policy.predict(observations, times, query_times, episode_ids=[201, 202])
print(f"loss={loss.item():.4f}; predicted action shape={tuple(actions.shape)}")
