import torch
from torch.autograd import Function
import numpy as np
from torch import jit


class Interpol(Function):
    @staticmethod
    def forward(ctx, grid_values, points, grid_x, grid_y, grid_z):
        # Perform trilinear interpolation
        x_indices = torch.searchsorted(grid_x, points[0], right=False) - 1
        y_indices = torch.searchsorted(grid_y, points[1], right=False) - 1
        z_indices = torch.searchsorted(grid_z, points[2], right=False) - 1

        x_weights = (points[0] - grid_x[x_indices]) / (
            grid_x[x_indices + 1] - grid_x[x_indices]
        )
        y_weights = (points[1] - grid_y[y_indices]) / (
            grid_y[y_indices + 1] - grid_y[y_indices]
        )
        z_weights = (points[2] - grid_z[z_indices]) / (
            grid_z[z_indices + 1] - grid_z[z_indices]
        )
        batch_indices = torch.arange(grid_values.shape[0], device=grid_values.device)
        c000 = grid_values[batch_indices, x_indices, y_indices, z_indices]
        c001 = grid_values[batch_indices, x_indices, y_indices, z_indices + 1]
        c010 = grid_values[batch_indices, x_indices, y_indices + 1, z_indices]
        c011 = grid_values[batch_indices, x_indices, y_indices + 1, z_indices + 1]
        c100 = grid_values[batch_indices, x_indices + 1, y_indices, z_indices]
        c101 = grid_values[batch_indices, x_indices + 1, y_indices, z_indices + 1]
        c110 = grid_values[batch_indices, x_indices + 1, y_indices + 1, z_indices]
        c111 = grid_values[batch_indices, x_indices + 1, y_indices + 1, z_indices + 1]

        result = (
            c000 * (1 - x_weights) * (1 - y_weights) * (1 - z_weights)
            + c001 * (1 - x_weights) * (1 - y_weights) * z_weights
            + c010 * (1 - x_weights) * y_weights * (1 - z_weights)
            + c011 * (1 - x_weights) * y_weights * z_weights
            + c100 * x_weights * (1 - y_weights) * (1 - z_weights)
            + c101 * x_weights * (1 - y_weights) * z_weights
            + c110 * x_weights * y_weights * (1 - z_weights)
            + c111 * x_weights * y_weights * z_weights
        )

        ctx.save_for_backward(
            grid_values,
            x_indices,
            y_indices,
            z_indices,
            x_weights,
            y_weights,
            z_weights,
        )

        return result

    @staticmethod
    @jit.script
    def backward_cuda(
        grad_output,
        grid_values,
        x_indices,
        y_indices,
        z_indices,
        x_weights,
        y_weights,
        z_weights,
    ):
        batch_indices = torch.arange(grid_values.shape[0], device=grid_values.device)
        grad_input = torch.zeros_like(grid_values)
        for i in range(8):
            dx = (i // 4) % 2
            dy = (i // 2) % 2
            dz = i % 2
            weight = (
                (1 - x_weights if dx == 0 else x_weights)
                * (1 - y_weights if dy == 0 else y_weights)
                * (1 - z_weights if dz == 0 else z_weights)
            )

            grad_weight = grad_output * weight

            grad_input[
                batch_indices, x_indices + dx, y_indices + dy, z_indices + dz
            ] += grad_weight

        return grad_input

    @staticmethod
    def backward(ctx, grad_output):
        (
            grid_values,
            x_indices,
            y_indices,
            z_indices,
            x_weights,
            y_weights,
            z_weights,
        ) = ctx.saved_tensors
        grad_values = Interpol.backward_cuda(
            grad_output,
            grid_values,
            x_indices,
            y_indices,
            z_indices,
            x_weights,
            y_weights,
            z_weights,
        )
        return (
            grad_values,
            None,
            None,
            None,
            None,
        )


if __name__ == "__main__":
    xpos = np.linspace(0, 2, 3)
    ypos = np.linspace(0, 4, 3)
    zpos = np.linspace(0, 6, 3)
    np.random.seed(0)
    values = np.random.rand(3, 3, 3)
    points = torch.rand(3, 400)
    xpos = torch.tensor(xpos, dtype=torch.float32)
    ypos = torch.tensor(ypos, dtype=torch.float32)
    zpos = torch.tensor(zpos, dtype=torch.float32)
    values = torch.tensor(values, requires_grad=True, dtype=torch.float32).repeat(
        points.shape[1], 1, 1, 1
    )

    # move all to cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xpos = xpos.to(device)
    ypos = ypos.to(device)
    zpos = zpos.to(device)
    values = values.to(device)
    points = points.to(device)

    interpolated_values = Interpol.apply(values, points, xpos, ypos, zpos)
    values.retain_grad()

    print(interpolated_values)
    interpolated_values.sum().backward()
    print(values.grad)
