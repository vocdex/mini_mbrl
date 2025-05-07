# VQ-VAE
## TLDR
VQ-VAE is a generative model that combines the strengths of VAEs and vector quantization. It uses discrete latent variables to improve reconstruction quality instead of imposing a Gaussian prior on the latent space. The prior is later learned using an autoregressive model (PixelCNN) to capture complex dependencies between codebook vectors. One can combine GANs with VQ-VAE to get the best of both worlds: high-quality images and discrete latent space. The model is trained using a combination of reconstruction loss, codebook loss, and commitment loss. The straight-through estimator allows for backpropagation through the quantization step, making it possible to train the model end-to-end.

### Introduction
Recently, VQ-VAEs are coming up every interesting work that I read : diffusion models(stable diffusion) and model-based RL (DreamerV2, Genie,) or video generation (VideoGPT). So, it's high time to get back to the basics and understand what makes VQ-VAEs better than normal VAEs in practice.

The paper "Neural Discrete Representation Learning"(van den Oord, Deepmind) introduced VQ-VAEs in 2017. The key innovation of VQ-VAE is the introduction of a vector quantization layer between the encoder and decode, which maps continuous representations to a discrete codebook of vectors.
### Why VQ-VAE?
Standard VAEs use continuous latent variables with a Gaussian prior, which can lead to:
- Posterior collapse (where the model ignores the latent space)
- Blurry outputs
- Difficulty modeling complex multimodal distributions

VQ-VAE addresses these issues by:
- using discrete latent variables
- avoiding the "averaging" effect of continuous distributions
- creating a more structured latent space

Encoder-Decoder layers are almost the same as in VAE. The vector quantization layer is the critical component. It:
- maintains a codebook of $K$ embedding vectors $e_k$
- for each encoded vector $z_e$, it finds the closest codebook vector by Euclidean distance
- replaces $z_e$ with the nearest codebook vector $e_k$: $z_q$
- so, $z_q=e_k$ where $k=\arg \min_j ||z_e-e_j||^2$
The decoder takes the quantized vector $z_q$ and attempts to reconstruct the original input.
### Loss
VQ-VAE loss consists of 3 terms:
1. **Reconstruction loss**: MSE for images and CE for categorical data
2. **Codebook loss**: moves the codebook vectors closer to the encoder outputs
3. **Commitment loss:** moves the encoder to commit to codebook vectors

$$L_{codebook}=||sg[z_e]-e||_2^2$$
$$L_{commit}=\beta ||z_e-sg[e]||_2^2$$
The stop gradient operator just stops the gradient flow through that parameter.
- In codebook loss, we want to move the codebook vectors (e) toward the encoder outputs ($z_e$), but we don't want to move $z_e$ toward $e$. The stop-gradient on $z_e$ prevents this.
- In commitment loss, we want to encourage the encoder to output embeddings $z_e$ close to codebook vectors ($e$), but we don't want to move $e$ toward $z_e$.
In **Pytorch**, stop gradient is implemented via **detach()** methods:
```python
z_q_sg=z_q.detach() # Stop gradient
```
### Straight-through-estimator
Stop gradient is part of Straight-through-estimator that does forward pass normally, but during the backward pass, it behaves as if we had simply allowed the gradients from earlier layer to flow through:
def straight_through_estimator(z_e, z_q):
```python
# Forward pass
z_e = encoder(x)
# Encoder output
# Quantization step (find nearest codebook vector)
distances = torch.sum((z_e.unsqueeze(1) - codebook.unsqueeze(0))**2, dim=2) indices = torch.argmin(distances, dim=1) # Non-differentiable!
z_q = codebook[indices] # Quantized vectors
# Straight-through estimator: z_q for forward, but gradient flows to z_e
z_pass = z_e + (z_q - z_e).detach()
# Continue forward pass x_recon = decoder(z_pass)
```

During backpropagation, this behaves as if we just $z_e$ directly.
### Why it works?
Despite being mathematically "incorrect," this approach is effective in practice because:
- In gradient descent, direction often matters more than precision
- The approximation introduces a form of implicit regularization
- As training progresses, encoder outputs naturally move closer to codebook vectors, making the approximation increasingly accurate


### KL term in VAE objective becomes 0
Remember in VAE, we have two terms in the objective:
$$Loss=\text{Reconstruction loss}+ D_{KL}(q(z|x)||p(z))$$
where $q(z|x)$ is the encoder and $p(z)$ is prior.
In VQ-VAE, the encoder outputs deterministic vectors, not distributions. The encoder output $z_e$ is mapped to the nearest codebook vector $z_q=e_k$. The prior is uniform over all codebook vectors.
Therefore, for a discrete distribution with a delta function at one point and a uniform prior, the KL divergence becomes a constant:
$$D_{KL}(q(z|x)||p(z))=\sum_{j=1}^Kq(z=z_k|x)\log \frac{q(z=z_k|x)}{p(z=z_k)} =1\times \log \frac{1}{1/K}$$
Since posterior is 0 for all $j\neq k$, all terms in the sum are zero except for $j=k$.
This allows VQ-VAE to focus on reconstruction quality without being constrained to follow a specific prior distribution in the latent space.
### Learning the Prior
Once VQ-VAE is fully trained, one can abandon the uniform prior imposed at training time and learn a new, updated prior $p(z)$ over the latents.

So, why do we need to abandon the uniform prior? Because in reality, certain codebook vectors are used more frequently than others for a given dataset. With a uniform prior, generating samples would mean randomly selecting codebook vectors with equal probability, which would produce poor results.

In the paper, they use PixelCNN autoregressive model to capture more complex dependencies between codebook vectors, treating them as discrete tokens.
$$p(z)=p(z_1)p(z_2|z_1)p(z_3|z_1,z_2)...$$

Actual training process looks is simple:
- take the original dataset used in VQ-VAE training
- pass each image through the trained VQ-VAE encoder and get its vector quantized latent index grids
- treat the dataset of latent index grids as training data for the PixelCNN
- maximize the log-likelihood of the observed latent index girds under PixelCNN model.


This outputs a trained PixelCNN models that captures the true distribution of $p(z)$ of the dataset (instead of uniform). This new prior allows to generate a new data sample that looks like it came from the original data distribution. To do so, we perform **ancestral sampling** to finally get a complete grid **z** of discrete latent indices. Finally, we can pass this grid indices to codebook quantization layer and get its quantized vectors, which will be passed to the decoder to get the final output.


### Training Tips

- Use high quality implemeentation VectorQuant from [vqtorch](https://github.com/vocdex/vqtorch) library.
- Using a large codebook size (e.g., 1024 or more) helps with reconstruction loss(maybe with quality too).
- Disentangled codebook vectors obtained using TSNE indicates that the model has learned meaningful representations. The model might still suffer from blurry reconstructions, but the codebook vectors are well-separated and this is usually not related to each other.
- Simple MSE loss is never sufficient for high-quality image generation. Use perceptual loss (e.g., VGG loss) or adversarial loss (GAN) to improve the quality of generated images.

