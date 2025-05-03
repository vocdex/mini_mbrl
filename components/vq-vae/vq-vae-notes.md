# VQ-VAE
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

VQ-VAE's success in diverse applications from image generation to reinforcement learning demonstrates the value of discrete latent representations in capturing complex data distributions and enabling more structured representations.
