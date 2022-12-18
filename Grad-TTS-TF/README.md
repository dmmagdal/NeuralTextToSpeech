# README

Description: Grad-TTS is a diffusion based neural text to speech model.

### 


Notes:
 * The original model was trained on a single Nvidia RTX 2080 Ti (11GB VRAM) for 1.7 million iterations (batch size 16). Current training speed on my 2060 SUPER (8GB VRAM) with the same batch size is ~10 epochs an hour (or 240 epochs/day). Given how there are 746 iterations per epoch, that means there needs to be 2,275 epochs (1.7M iters X 746 iters/epoch X 240 epochs/day) or rather 9.5 days of training on this device. 