import torch
import matplotlib.pyplot as plt
import math

def closest_factors(num): 
    sq = int(math.sqrt(num)) + 1
    for i in range(sq, 1, -1): 
        if num % i == 0: 
            return i, num // i
    return 1, num

def plot_image_grid(tensor, title="Image Grid"):   
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (N, 3, H, W) for RGB images.")

    f1, f2 = closest_factors(tensor.shape[0])

    fig, axes = plt.subplots(f1, f2, figsize=(f2 * 2, f1 * 2))
    fig.suptitle(title)

    axes = axes.flatten() if tensor.shape[0] > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < tensor.shape[0]:
            img = tensor[i].detach().cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_tensor_matrix(tensor, title="Tensor Matrix", cmap="viridis"):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if tensor.dim() != 2:
        raise ValueError("Only 2D tensors can be plotted as matrices.")
    plt.imshow(tensor.detach().cpu().numpy(), cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()


def generate_betas(beta_0, beta_T, num_steps): 
    delta = beta_T - beta_0 
    return [delta*i/(num_steps-1) + beta_0 for i in range(num_steps)] 

def generate_alpha(beta_0, beta_T, num_steps): 
    betas = generate_betas(beta_0, beta_T, num_steps)
    alphas = [1-b for b in betas]
    return alphas

def generate_alpha_bar(beta_0, beta_T, num_steps): 
    alphas = generate_alpha(beta_0, beta_T, num_steps)
    alpha_bar = [alphas[0]]
    for i in range(1, len(alphas)): alpha_bar.append(alpha_bar[i-1] * alphas[i])
    return alpha_bar