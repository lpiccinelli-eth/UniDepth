import warnings
from typing import Union

import torch

try:
    from unidepth.ops.knn import knn_points
except ImportError as e:
    warnings.warn(
        "!! To run evaluation you need KNN. Please compile KNN: "
        "`cd unidepth/ops/knn with && bash compile.sh`."
    )
    knn_points = lambda x : x


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: torch.Tensor,
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if points.ndim != 3:
        raise ValueError("Expected points to be of shape (N, P, D)")
    X = points
    if lengths is not None and (lengths.ndim != 1 or lengths.shape[0] != X.shape[0]):
        raise ValueError("Expected lengths to be of shape (N,)")
    if lengths is None:
        lengths = torch.full(
            (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
        )
    if normals is not None and normals.ndim != 3:
        raise ValueError("Expected normals to be of shape (N, P, 3")

    return X, lengths, normals


class ChamferDistance(torch.nn.Module):
    def forward(
        self,
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
    ):
        """
        Chamfer distance between two pointclouds x and y.

        Args:
            x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
                a batch of point clouds with at most P1 points in each batch element,
                batch size N and feature dimension D.
            y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
                a batch of point clouds with at most P2 points in each batch element,
                batch size N and feature dimension D.
            x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
                cloud in x.
            y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
                cloud in x.
            x_normals: Optional FloatTensor of shape (N, P1, D).
            y_normals: Optional FloatTensor of shape (N, P2, D).
            weights: Optional FloatTensor of shape (N,) giving weights for
                batch elements for reduction operation.
            batch_reduction: Reduction operation to apply for the loss across the
                batch, can be one of ["mean", "sum"] or None.
            point_reduction: Reduction operation to apply for the loss across the
                points, can be one of ["mean", "sum"].

        Returns:
            2-element tuple containing

            - **loss**: Tensor giving the reduced distance between the pointclouds
              in x and the pointclouds in y.
            - **loss_normals**: Tensor giving the reduced cosine distance of normals
              between pointclouds in x and pointclouds in y. Returns None if
              x_normals and y_normals are None.
        """
        _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

        x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
        y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

        return_normals = x_normals is not None and y_normals is not None

        N, P1, D = x.shape
        P2 = y.shape[1]

        # Check if inputs are heterogeneous and create a lengths mask.
        is_x_heterogeneous = (x_lengths != P1).any()
        is_y_heterogeneous = (y_lengths != P2).any()
        x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
        )  # shape [N, P1]
        y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
        )  # shape [N, P2]

        if y.shape[0] != N or y.shape[2] != D:
            raise ValueError("y does not have the correct shape.")
        if weights is not None:
            if weights.size(0) != N:
                raise ValueError("weights must be of shape (N,).")
            if not (weights >= 0).all():
                raise ValueError("weights cannot be negative.")
            if weights.sum() == 0.0:
                weights = weights.view(N, 1)
                if batch_reduction in ["mean", "sum"]:
                    return (
                        (x.sum((1, 2)) * weights).sum() * 0.0,
                        (x.sum((1, 2)) * weights).sum() * 0.0,
                    )
                return (
                    (x.sum((1, 2)) * weights) * 0.0,
                    (x.sum((1, 2)) * weights) * 0.0,
                )

        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
        y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

        cham_x = x_nn.dists[..., 0]  # (N, P1)
        cham_y = y_nn.dists[..., 0]  # (N, P2)

        if is_x_heterogeneous:
            cham_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_y[y_mask] = 0.0

        if weights is not None:
            cham_x *= weights.view(N, 1)
            cham_y *= weights.view(N, 1)

        return cham_x, cham_y, x_nn.idx[..., -1], y_nn.idx[..., -1]
