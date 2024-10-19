### VAE
To reproduce the results of the VAE model, run the following command:
```bash
python vae.py --train --test --plot
```
This will train the VAE model on the training data, test it on the test data, and plot the results and save them in the `plots` directory.

To record a comparison video between the original and reconstructed images, run the following command:
```bash
python compare_vae_gym.py --record
```
This will allow you to control the environment with keyboard inputs and record a video of the side-by-side comparison.
