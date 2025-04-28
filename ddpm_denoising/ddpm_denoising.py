import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

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

def closest_factors(num): 
    sq = int(math.sqrt(num))+1
    for i in range(sq, 1, -1): 
        if(num % i == 0): 
            return i, num//i
    return 1,num

def plot_image_grid(tensor, title="Image Grid", cmap="gray"):   
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    # if tensor.shape != (16, 28, 28):
    #     raise ValueError("Input tensor must have shape (16, 28, 28).")
    f1, f2 = closest_factors(tensor.shape[0])


    fig, axes = plt.subplots(f1, f2)
    fig.suptitle(title)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(tensor[i].detach().cpu().numpy(), cmap=cmap)
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # leave space for the title
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

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total

if __name__ == "__main__":
    model_file = "../model_testing/noise_predictor_guided.pth"
    mym = torch.load(model_file, weights_only=False)

    beta_0 = 0.0001
    beta_T = 0.05
    num_steps = 256
    a_bar = generate_alpha_bar(beta_0, beta_T, num_steps=num_steps)
    alpha = generate_alpha(beta_0, beta_T, num_steps=num_steps)

    def normalize_zero_mean_unit_var(x, eps=1e-8):
        mean = x.mean()
        std = x.std(unbiased=False)
        return (x - mean) / (std + eps)

    mym.eval()
    count_parameters(mym)

    X_t = torch.randn(10,1,28,28).cuda()

    T = num_steps
    storage = torch.zeros((T,28,28))

    label = torch.tensor(range(10))

    guidance_scale = 2
    
    with torch.no_grad():
        for t in tqdm(range(T-1,-1,-1)): 
            Z = torch.randn_like(X_t)
            sigma_t = 0.5 * (1 - alpha[t]) ** 0.5 
            if(t == 0): sigma_t = 0
            
            time = t * torch.ones(10).cuda()
            guided = torch.ones(10).cuda()

            X_in = torch.concat([X_t, X_t], dim=0)
            time_in = torch.concat([time, time], dim=0)
            label_in = torch.concat([label, label], dim=0)
            guided_in = torch.concat([guided, 0*guided], dim=0)

            noise = mym(X_in, time_in, label_in, guided_in)
            noise_guided = noise[:10]
            noise_unguided = noise[10:]

            noise_final = noise_unguided + guidance_scale * (noise_guided - noise_unguided)

            # PREDICT NEXT TIME STEP 
            X_t = (1/math.sqrt(alpha[t]))*(X_t - ((1-alpha[t])/(math.sqrt(1-a_bar[t]))) * noise_final) + sigma_t * Z

            storage[T-t-1] = X_t[-1].cpu()

    plot_image_grid(storage[::8])
    plot_image_grid(X_t[:,0])
