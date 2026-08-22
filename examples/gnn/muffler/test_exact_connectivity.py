from pathlib import Path
import sys
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "foampilot" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cfd_gnn import UniversalGraphExtractor  # noqa: E402


def main():
    grid = pv.ImageData(dimensions=(3, 2, 2), spacing=(1.0, 1.0, 1.0))
    extractor = UniversalGraphExtractor(Path("."))
    extractor._cell_mesh = grid
    extractor._node_positions = grid.cell_centers().points
    extractor._spatial_dim = 3
    edge_index, edge_features = extractor._extract_edge_features()
    edges = {tuple(e) for e in edge_index.t().tolist()}
    assert edges == {(0, 1), (1, 0)}, edges
    assert edge_features.shape == (2, 4), edge_features.shape
    print("exact_connectivity_test=PASS")
    print("edges=", sorted(edges))
    print("edge_features_shape=", tuple(edge_features.shape))


if __name__ == "__main__":
    main()
