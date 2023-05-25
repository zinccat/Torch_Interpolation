import torch
from torch.autograd import Function
import numpy as np
from torch import jit


class Interpol(Function):
    @staticmethod
    def forward(ctx, grid_values, points, grid):
        # Perform trilinear interpolation
        n_dim = points.shape[0]
        indices = torch.zeros_like(points, dtype=torch.long)
        weights = torch.zeros_like(points)
        for i in range(n_dim):
            indices[i] = torch.searchsorted(grid[i], points[i], right=False) - 1
        for i in range(n_dim):
            weights[i] = (points[i] - grid[i][indices[i]]) / (
                grid[i][indices[i] + 1] - grid[i][indices[i]]
            )
        batch_indices = torch.arange(grid_values.shape[0], device=grid_values.device)

        pn_dim = 2**n_dim
        c = torch.zeros((pn_dim, points.shape[1]), device=points.device)
        for i in range(pn_dim):
            dx = i // 4
            dy = (i % 4) // 2
            dz = i % 2
            c[i] = grid_values[
                batch_indices, indices[0] + dx, indices[1] + dy, indices[2] + dz
            ]
        result = torch.zeros((points.shape[1],), device=points.device)
        for i in range(pn_dim):
            result += c[i] * (
                (1 - weights[0] if i // 4 == 0 else weights[0])
                * (1 - weights[1] if (i % 4) // 2 == 0 else weights[1])
                * (1 - weights[2] if i % 2 == 0 else weights[2])
            )
        ctx.save_for_backward(
            grid_values,
            indices,
            weights,
        )
        return result

    @staticmethod
    @jit.script
    def backward_cuda(
        grad_output,
        grid_values,
        indices,
        weights,
    ):
        batch_indices = torch.arange(grid_values.shape[0], device=grid_values.device)
        grad_input = torch.zeros_like(grid_values)
        for i in range(8):
            dx = (i // 4) % 2
            dy = (i // 2) % 2
            dz = i % 2
            weight = (
                (1 - weights[0] if dx == 0 else weights[0])
                * (1 - weights[1] if dy == 0 else weights[1])
                * (1 - weights[2] if dz == 0 else weights[2])
            )
            grad_input[
                batch_indices,
                indices[0] + dx,
                indices[1] + dy,
                indices[2] + dz,
            ] += (
                grad_output * weight
            )

        return grad_input

    @staticmethod
    def backward(ctx, grad_output):
        grid_values, indices, weights = ctx.saved_tensors
        grad_values = Interpol.backward_cuda(grad_output, grid_values, indices, weights)
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
    torch.random.manual_seed(0)
    values = np.random.rand(3, 3, 3)
    points = torch.rand(3, 400)
    xpos = torch.tensor(xpos, dtype=torch.float32)
    ypos = torch.tensor(ypos, dtype=torch.float32)
    zpos = torch.tensor(zpos, dtype=torch.float32)
    values = torch.tensor(values, requires_grad=True, dtype=torch.float32).repeat(
        points.shape[1], 1, 1, 1
    )
    # print(values.shape)
    # exit(0)

    # move all to cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xpos = xpos.to(device)
    ypos = ypos.to(device)
    zpos = zpos.to(device)
    grid = [xpos, ypos, zpos]
    values = values.to(device)
    points = points.to(device)

    interpolated_values = Interpol.apply(values, points, grid)
    values.retain_grad()

    print(interpolated_values.shape)
    interpolated_values.sum().backward()
    print(values.grad)
