# losses.py
# Implement all loss functions for the GAN models.


import tensorflow as tf


def feature_loss(fmap_r, fmap_g):
	loss = 0

	for dr, dg in zip(fmap_r, fmap_g):
		for rl, gl in zip(dr, dg):
			loss += tf.math.reduce_mean(
				tf.math.abs(rl - gl)
			)

	return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
	loss = 0
	r_losses = []
	g_losses = []

	for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
		r_loss = tf.math.reduce_mean((1 - dr) ** 2)
		g_loss = tf.math.reduce_mean(dg ** 2)
		loss += (r_loss + g_loss)
		r_losses.append(r_loss)
		g_losses.append(g_loss)

	return loss, r_losses, g_losses


def generator_loss(disc_outputs):
	loss = 0
	gen_losses = []
	
	for dg in disc_outputs:
		l = tf.math.reduce_mean((1 - dg) ** 2)
		gen_losses.append(l)
		loss += l
	
	return loss, gen_losses