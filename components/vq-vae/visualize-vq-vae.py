"""Utilities for VQ-VAE latent space visualization and manipulation"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from vqvae_cifar10 import VQVAE, Box, Config, get_data_loaders


def encode_dataset(model, data_loader, device, num_batches=None):
    """
    Encode a dataset and collect latent representations and indices

    Args:
        model: Trained VQ-VAE model
        data_loader: DataLoader for the dataset
        device: Device to run encoding on
        num_batches: Number of batches to encode (None = all)

    Returns:
        all_indices: Tensor of all codebook indices [N, H*W]
        all_encodings: Tensor of all latent encodings [N, D, H, W]
        all_images: Tensor of all original images [N, C, H, W]
    """
    model.eval()
    all_indices = []
    all_encodings = []
    all_images = []

    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break

            data = data.to(device)
            all_images.append(data.cpu())

            # Encode data
            z_e = model.encoder(data)
            z_q, _, indices = model.vq_layer(z_e)

            # Reshape indices to [B, H*W]
            b, _, h, w = z_e.shape
            indices = indices.view(b, h * w)

            all_indices.append(indices.cpu())
            all_encodings.append(z_q.cpu())

    # Concatenate all batches
    all_indices = torch.cat(all_indices, dim=0)
    all_encodings = torch.cat(all_encodings, dim=0)
    all_images = torch.cat(all_images, dim=0)

    return all_indices, all_encodings, all_images


def visualize_codebook_usage(indices, num_embeddings, figsize=(10, 4)):
    """
    Visualize how frequently each codebook vector is used

    Args:
        indices: Tensor of codebook indices [N, H*W]
        num_embeddings: Number of embeddings in the codebook
        figsize: Figure size for the plot
    """
    # Count frequency of each index
    flat_indices = indices.flatten().numpy()
    histogram = np.bincount(flat_indices, minlength=num_embeddings)

    # Plot histogram
    plt.figure(figsize=figsize)
    plt.bar(np.arange(num_embeddings), histogram)
    plt.xlabel("Codebook Index")
    plt.ylabel("Frequency")
    plt.title("Codebook Usage Frequency")
    plt.tight_layout()
    plt.show()

    # Print statistics
    used_codes = (histogram > 0).sum()
    print(f"Codebook usage: {used_codes}/{num_embeddings} vectors used ({used_codes/num_embeddings:.2%})")
    print(f"Most common index: {histogram.argmax()} (used {histogram.max()} times)")
    print(f"Least common index: {histogram.argmin()} (used {histogram.min()} times)")


def visualize_latent_space(encodings, indices, num_samples=1000, method="tsne", figsize=(10, 10)):
    """
    Visualize the latent space using dimensionality reduction

    Args:
        encodings: Tensor of latent encodings [N, D, H, W]
        indices: Tensor of codebook indices [N, H*W]
        num_samples: Number of samples to use for visualization
        method: Dimensionality reduction method ('tsne' or 'pca')
        figsize: Figure size for the plot
    """
    # Get a subset of the encodings
    if encodings.shape[0] > num_samples:
        idx = np.random.choice(encodings.shape[0], num_samples, replace=False)
        encodings_subset = encodings[idx]
        indices_subset = indices[idx]
    else:
        encodings_subset = encodings
        indices_subset = indices

    # Reshape encodings to [N*H*W, D]
    n, d, h, w = encodings_subset.shape
    flat_encodings = encodings_subset.permute(0, 2, 3, 1).reshape(-1, d)

    # Reshape indices to [N*H*W]
    flat_indices = indices_subset.reshape(-1)

    # Sample a subset of points for visualization
    if flat_encodings.shape[0] > 10000:
        idx = np.random.choice(flat_encodings.shape[0], 10000, replace=False)
        flat_encodings = flat_encodings[idx]
        flat_indices = flat_indices[idx]

    # Apply dimensionality reduction
    print(f"Applying {method.upper()} to {flat_encodings.shape[0]} points...")
    if method.lower() == "tsne":
        latent_2d = TSNE(n_components=2, random_state=42).fit_transform(flat_encodings.numpy())
    else:  # PCA
        latent_2d = PCA(n_components=2, random_state=42).fit_transform(flat_encodings.numpy())

    # Plot results
    plt.figure(figsize=figsize)
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=flat_indices.numpy(), cmap="tab20", alpha=0.5, s=5)
    plt.colorbar(label="Codebook Index")
    plt.title(f"VQ-VAE Latent Space Visualization ({method.upper()})")
    plt.tight_layout()
    plt.show()


def visualize_latent_grid(model, device, grid_size=8, latent_size=(4, 4)):
    """
    Generate images by sampling from a grid in the latent space

    Args:
        model: Trained VQ-VAE model
        device: Device to run model on
        grid_size: Size of the grid (grid_size x grid_size)
        latent_size: Size of the latent space (H, W)
    """
    model.eval()

    # Create a grid of indices
    h, w = latent_size
    n_embeddings = model.vq_layer.num_embeddings

    # Use two different codebook indices for row and column
    code1 = torch.randint(0, n_embeddings, (1,)).item()
    code2 = torch.randint(0, n_embeddings, (1,)).item()
    while code2 == code1:
        code2 = torch.randint(0, n_embeddings, (1,)).item()

    # Create grid of images
    all_imgs = []

    with torch.no_grad():
        for i in range(grid_size):
            row_imgs = []
            for j in range(grid_size):
                # Create latent representation filled with code1
                indices = torch.ones(1, h * w, dtype=torch.long) * code1

                # Interpolate between code1 and code2
                ratio_i = i / (grid_size - 1)
                ratio_j = j / (grid_size - 1)

                # Set some indices to code2 based on position in grid
                num_code2 = int((ratio_i + ratio_j) / 2 * h * w)
                rand_pos = torch.randperm(h * w)[:num_code2]
                indices[0, rand_pos] = code2

                # Decode the indices
                z_q = model.vq_layer.embedding(indices).view(1, -1, h, w)
                z_q = z_q.permute(0, 1, 2, 3)  # Adjust dimensions as needed

                # Generate image
                img = model.decoder(z_q)
                row_imgs.append(img)

            # Concatenate images in this row
            row_grid = torch.cat(row_imgs, dim=0)
            all_imgs.append(row_grid)

        # Concatenate all rows
        grid = torch.cat(all_imgs, dim=0)

    # Visualize the grid
    grid = (grid * 0.5) + 0.5  # Unnormalize
    grid = torch.clamp(grid, 0, 1)
    grid_img = torchvision.utils.make_grid(grid, nrow=grid_size)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(f"Latent Space Interpolation (Code {code1} → Code {code2})")
    plt.axis("off")
    plt.show()


def visualize_spatial_codes(indices, image_shape, cmap="tab20", figsize=(8, 8)):
    """
    Visualize spatial distribution of codebook indices

    Args:
        indices: Tensor of codebook indices [N, H*W]
        image_shape: Shape of the latent images (H, W)
        cmap: Colormap for visualization
        figsize: Figure size
    """
    # Take a random sample
    idx = np.random.randint(0, indices.shape[0], 16)
    images = indices[idx]

    # Reshape to spatial dimensions
    h, w = image_shape
    images = images.reshape(-1, h, w)

    # Create plot
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, images)):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.suptitle("Spatial Distribution of Codebook Indices")
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()


def latent_arithmetic(model, data_loader, device, n_samples=5):
    """
    Perform latent arithmetic operations

    Args:
        model: Trained VQ-VAE model
        data_loader: DataLoader for the dataset
        device: Device to run model on
        n_samples: Number of samples to use
    """
    model.eval()

    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images[:n_samples].to(device)

    # Encode images
    with torch.no_grad():
        z_e = model.encoder(images)
        z_q, _, indices = model.vq_layer(z_e)

        # Get original reconstructions
        recons = model.decoder(z_q)

        # Perform some operations on latent codes
        b, c, h, w = z_q.shape

        # 1. Swap top and bottom half
        z_q_swap = z_q.clone()
        z_q_swap[:, :, : h // 2, :] = z_q[:, :, h // 2 :, :]
        z_q_swap[:, :, h // 2 :, :] = z_q[:, :, : h // 2, :]
        recons_swap = model.decoder(z_q_swap)

        # 2. Mirror left-right
        z_q_mirror = torch.flip(z_q, [3])  # Flip along width dimension
        recons_mirror = model.decoder(z_q_mirror)

        # 3. Mix two images (A + B - A)
        if n_samples >= 2:
            z_q_mix = z_q[1:2].clone()  # Take second image
            z_q_mix[:, :, h // 2 :, :] = z_q[0:1, :, h // 2 :, :]  # Replace bottom half with first image
            recons_mix = model.decoder(z_q_mix)
        else:
            recons_mix = None

        # 4. Average of all latent codes
        z_q_avg = torch.mean(z_q, dim=0, keepdim=True).repeat(b, 1, 1, 1)
        recons_avg = model.decoder(z_q_avg)

    # Visualize results
    fig, axes = plt.subplots(n_samples, 5, figsize=(15, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    # Unnormalize
    images = (images * 0.5) + 0.5
    recons = (recons * 0.5) + 0.5
    recons_swap = (recons_swap * 0.5) + 0.5
    recons_mirror = (recons_mirror * 0.5) + 0.5
    recons_avg = (recons_avg * 0.5) + 0.5
    if recons_mix is not None:
        recons_mix = (recons_mix * 0.5) + 0.5

    # Clamp to valid range
    images = torch.clamp(images, 0, 1)
    recons = torch.clamp(recons, 0, 1)
    recons_swap = torch.clamp(recons_swap, 0, 1)
    recons_mirror = torch.clamp(recons_mirror, 0, 1)
    recons_avg = torch.clamp(recons_avg, 0, 1)
    if recons_mix is not None:
        recons_mix = torch.clamp(recons_mix, 0, 1)

    titles = [
        "Original",
        "Reconstruction",
        "Top-Bottom Swap",
        "Mirror",
        "Mixed" if recons_mix is not None else "Average",
    ]

    for i in range(n_samples):
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
        axes[i, 1].imshow(recons[i].cpu().permute(1, 2, 0))
        axes[i, 2].imshow(recons_swap[i].cpu().permute(1, 2, 0))
        axes[i, 3].imshow(recons_mirror[i].cpu().permute(1, 2, 0))

        if recons_mix is not None and i == 0:
            axes[i, 4].imshow(recons_mix[0].cpu().permute(1, 2, 0))
            axes[i, 4].set_title("1 + 2 (bottom half)")
        else:
            axes[i, 4].imshow(recons_avg[i].cpu().permute(1, 2, 0))

        for j in range(5):
            axes[i, j].set_title(titles[j] if i == 0 else "")
            axes[i, j].axis("off")

    plt.suptitle("Latent Space Arithmetic Operations")
    plt.tight_layout()
    plt.show()


def visualize_codebook_vectors(model, grid_size=16):
    """
    Visualize the learned codebook vectors in the VQ-VAE

    Args:
        model: Trained VQ-VAE model
        grid_size: Size of the grid to display (sqrt of number of vectors to show)
    """
    # Get the embedding weights
    embedding = model.vq_layer.embedding.weight.data.cpu()  # Shape: [K, D]

    # Determine how many vectors to visualize
    n_vectors = min(grid_size * grid_size, embedding.shape[0])
    grid_size = int(np.sqrt(n_vectors))

    # Normalize the embeddings for visualization
    embedding_norm = embedding[:n_vectors]
    embedding_norm = (embedding_norm - embedding_norm.min()) / (embedding_norm.max() - embedding_norm.min())

    # Reshape to square grid for visualization
    grid_vectors = embedding_norm.view(grid_size, grid_size, -1)

    # PCA to project to 3D for RGB visualization
    flat_vectors = embedding_norm.numpy()
    pca = PCA(n_components=3)
    vectors_rgb = pca.fit_transform(flat_vectors)
    vectors_rgb = (vectors_rgb - vectors_rgb.min()) / (vectors_rgb.max() - vectors_rgb.min())
    vectors_rgb = vectors_rgb.reshape(grid_size, grid_size, 3)

    # Plot the grid
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Plot 1: Heatmap of first two principal components
    heatmap = ax[0].imshow(grid_vectors[:, :, 0])
    ax[0].set_title("First Component of Codebook Vectors")
    ax[0].axis("off")
    plt.colorbar(heatmap, ax=ax[0])

    # Plot 2: RGB visualization
    ax[1].imshow(vectors_rgb)
    ax[1].set_title("PCA RGB Visualization of Codebook Vectors")
    ax[1].axis("off")

    plt.suptitle(f"Visualization of {n_vectors} Codebook Vectors")
    plt.tight_layout()
    plt.show()


def interactive_latent_exploration(model, device, grid_size=10):
    """
    Interactive exploration of latent space by generating images from manually specified codes

    Args:
        model: Trained VQ-VAE model
        device: Device to run model on
        grid_size: Size of the latent space grid (H=W)
    """
    model.eval()

    # Get model parameters
    num_embeddings = model.vq_layer.num_embeddings

    print(f"Interactive Latent Space Explorer")
    print(f"===================================")
    print(f"Codebook size: {num_embeddings} vectors")
    print(f"Latent grid size: {grid_size}x{grid_size}")
    print()
    print("Options:")
    print("1. Generate image with random codes")
    print("2. Generate image with specified code pattern")
    print("3. Generate image with code grid")
    print("4. Exit")

    while True:
        choice = input("\nEnter choice (1-4): ")

        if choice == "1":
            # Random codes
            indices = torch.randint(0, num_embeddings, (1, grid_size * grid_size))

        elif choice == "2":
            # Specified pattern
            pattern_type = input("Enter pattern type (uniform, gradient, checkerboard): ")

            if pattern_type == "uniform":
                code = int(input(f"Enter code index (0-{num_embeddings-1}): "))
                indices = torch.ones(1, grid_size * grid_size, dtype=torch.long) * code

            elif pattern_type == "gradient":
                code1 = int(input(f"Enter first code index (0-{num_embeddings-1}): "))
                code2 = int(input(f"Enter second code index (0-{num_embeddings-1}): "))

                indices = torch.ones(1, grid_size * grid_size, dtype=torch.long) * code1
                for i in range(grid_size):
                    ratio = i / (grid_size - 1)
                    idx_start = i * grid_size
                    idx_end = (i + 1) * grid_size
                    num_code2 = int(ratio * grid_size)

                    if num_code2 > 0:
                        indices[0, idx_start : idx_start + num_code2] = code2

            elif pattern_type == "checkerboard":
                code1 = int(input(f"Enter first code index (0-{num_embeddings-1}): "))
                code2 = int(input(f"Enter second code index (0-{num_embeddings-1}): "))

                indices = torch.ones(1, grid_size * grid_size, dtype=torch.long) * code1
                for i in range(grid_size):
                    for j in range(grid_size):
                        if (i + j) % 2 == 1:
                            indices[0, i * grid_size + j] = code2
            else:
                print("Invalid pattern type")
                continue

        elif choice == "3":
            # Code grid
            print("Enter code for each position (or 'r' for random):")
            indices = torch.zeros(1, grid_size * grid_size, dtype=torch.long)

            for i in range(grid_size):
                row_input = input(f"Row {i+1}: ")
                codes = row_input.split()

                if len(codes) != grid_size:
                    print(f"Expected {grid_size} values, got {len(codes)}. Try again.")
                    i -= 1
                    continue

                for j, code in enumerate(codes):
                    if code.lower() == "r":
                        indices[0, i * grid_size + j] = torch.randint(0, num_embeddings, (1,))
                    else:
                        try:
                            code_val = int(code)
                            if 0 <= code_val < num_embeddings:
                                indices[0, i * grid_size + j] = code_val
                            else:
                                print(f"Code {code_val} out of range. Using random.")
                                indices[0, i * grid_size + j] = torch.randint(0, num_embeddings, (1,))
                        except ValueError:
                            print(f"Invalid code '{code}'. Using random.")
                            indices[0, i * grid_size + j] = torch.randint(0, num_embeddings, (1,))

        elif choice == "4":
            # Exit
            break

        else:
            print("Invalid choice")
            continue

        # Generate and display image
        with torch.no_grad():
            # Reshape indices to match spatial dimensions
            indices_spatial = indices.reshape(1, grid_size, grid_size)

            z_q = model.vq_layer.embedding(indices.to(device)).view(1, -1, grid_size, grid_size)
            z_q = z_q.permute(0, 1, 2, 3)  # [B, D, H, W]

            # Generate image
            img = model.decoder(z_q)

            # Convert to displayable format
            img = (img * 0.5) + 0.5  # Unnormalize
            img = torch.clamp(img, 0, 1)

            # Also display the code pattern
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Display generated image
            axes[0].imshow(img[0].cpu().permute(1, 2, 0))
            axes[0].set_title("Generated Image")
            axes[0].axis("off")

            # Display code pattern
            code_img = axes[1].imshow(indices_spatial[0].cpu(), cmap="tab20")
            plt.colorbar(code_img, ax=axes[1], label="Codebook Index")
            axes[1].set_title("Code Pattern")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()


def example_usage():
    """Example of how to use the visualization functions with a trained model"""

    config = Config()
    cfg = Box(config.to_dict())

    model = VQVAE(cfg).to(cfg.device)

    # Try to load a saved model, but continue with randomly initialized weights if file not found
    model_path = "./models/vqvae_best.pt"
    try:
        model.load(model_path)
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"No trained model found at {model_path}. Using randomly initialized weights.")
        print("NOTE: Results will not be meaningful without a trained model.")
        print("Train the model first or specify the correct path to a saved model.")

    train_loader, test_loader = get_data_loaders(cfg)

    print("Encoding a subset of the dataset...")
    indices, encodings, images = encode_dataset(model, test_loader, cfg.device, num_batches=10)

    # Visualize codebook usage
    print("\nAnalyzing codebook usage patterns...")
    visualize_codebook_usage(indices, cfg.num_embeddings)

    # Visualize latent space
    print("\nVisualizing latent space using dimensionality reduction...")
    visualize_latent_space(encodings, indices, num_samples=500)

    # Visualize spatial distribution of codes
    print("\nVisualizing spatial arrangement of latent codes...")
    h, w = cfg.image_size // 8, cfg.image_size // 8  # Latent space dimensions
    visualize_spatial_codes(indices, (h, w))

    # Latent arithmetic
    print("\nPerforming operations in latent space...")
    latent_arithmetic(model, test_loader, cfg.device, n_samples=3)

    # Visualize codebook vectors
    print("\nVisualizing the learned codebook vectors...")
    visualize_codebook_vectors(model)

    # Interactive exploration
    print("\nStarting interactive latent space exploration...")
    interactive_latent_exploration(model, cfg.device)


if __name__ == "__main__":
    example_usage()
