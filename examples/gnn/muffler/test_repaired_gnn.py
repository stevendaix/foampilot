from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "foampilot" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_cfd_gnn import (  # noqa: E402
    GNNArchitectureConfig,
    PhysicsConfig,
    UniversalGNN,
    UniversalGraphConv,
)


def main():
    torch.manual_seed(7)
    conv = UniversalGraphConv(8, 8, use_attention=False, aggregation="mean")
    x = torch.randn(5, 8, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    edge_features = torch.randn(edge_index.shape[1], 4)
    y, _ = conv(x, edge_index, edge_features)
    assert y.shape == x.shape
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None

    cfg = GNNArchitectureConfig(
        hidden_dim=32, n_layers=2, use_attention=False,
        output_variables=["p", "Ux", "Uy", "Uz"],
        include_node_position=True,
    )
    model = UniversalGNN(cfg, PhysicsConfig(), spatial_dim=3, input_dim=11)
    node_features = torch.randn(5, 11)
    pred = model(node_features, edge_index, edge_features)
    assert pred["p"].shape == (5, 1)
    assert pred["U"].shape == (5, 3)
    total = sum(v.square().mean() for k, v in pred.items() if k in {"p", "U"})
    total.backward()
    print("repaired_gnn_test=PASS")
    print("conv_edges=", edge_index.shape[1])
    print("outputs=", {k: tuple(v.shape) for k, v in pred.items() if k in {"p", "U"}})


if __name__ == "__main__":
    main()
