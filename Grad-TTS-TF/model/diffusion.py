# diffusion.py


import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
# from .group_normalizations import GroupNormalization
import tensorflow_addons as tfa
from tensorflow_addons.layers import GroupNormalization
from .identity import Identity
from einops import rearrange


class Mish(layers.Layer):
	def __init__(self):
		super(Mish, self).__init__()
		# self.softplus = layers.Activation('softplus')


	def call(self, x):
		# return x * tf.math.tanh(self.softplus(x))
		return x * tf.math.tanh(tf.math.softplus(x))


class UpSample(layers.Layer):
	def __init__(self, dim):
		super(UpSample, self).__init__()
		self.conv = layers.Conv2DTranspose(dim, 4, 2, "same")


	def call(self, x):
		return self.conv(x)


class DownSample(layers.Layer):
	def __init__(self, dim):
		super(DownSample, self).__init__()
		self.conv = layers.Conv2D(dim, 3, 2, "same")


	def call(self, x):
		return self.conv(x)


class ReZero(layers.Layer):
	def __init__(self, fn):
		super(ReZero, self).__init__()
		self.fn = fn
		self.g = self.add_weight("g", 1, initializer="zeros", trainable=True)


	def call(self, x):
		return self.fn(x) * self.g


class Block(layers.Layer):
	def __init__(self, dim_out, groups=8):
		super(Block, self).__init__()
		self.block = Sequential([
			layers.Conv2D(dim_out, 3, padding="same"),
			GroupNormalization(groups=groups),
			Mish()
		])


	def call(self, x, mask):
		print(f"shape {(x * mask).shape}")
		output = self.block(x * mask) # Original
		print(f"output shape {output.shape}")
		print(f"output shape {output.shape}")
		return output * mask


class ResnetBlock(layers.Layer):
	def __init__(self, dim, dim_out, time_emb_dim, groups=8):
		super(ResnetBlock, self).__init__()
		self.mlp = Sequential([
			Mish(),
			layers.Dense(dim_out)
		])
		self.block1 = Block(dim_out,  groups=groups)
		self.block2 = Block(dim_out,  groups=groups)

		if dim != dim_out:
			self.res_conv = layers.Conv2D(dim_out, 1)
		else:
			self.res_conv = Identity()


	def call(self, x, mask, time_emb):
		h = self.block1(x, mask)
		print(f"h block1 {h}, shape {h.shape}")
		temp = tf.expand_dims(tf.expand_dims(self.mlp(time_emb), 1), 1)
		print(f"time_emb mlp {temp}, shape {temp.shape}")
		h += tf.expand_dims(
			# tf.expand_dims(self.mlp(time_emb), -1), -1 # Original
			# tf.expand_dims(self.mlp(time_emb), -1), 1 # failed
			# tf.expand_dims(self.mlp(time_emb), 1), -1 # failed
			tf.expand_dims(self.mlp(time_emb), 1), 1
		)
		print(f"h mlp {h}, shape {h.shape}")
		h = self.block2(h, mask)
		print(f"h block2 {h}, shape {h.shape}")
		output = h + self.res_conv(x * mask)
		print(f"output h + res_conv {output}, shape {output.shape}")
		return output


class LinearAttention(layers.Layer):
	def __init__(self, dim, heads=4, dim_head=32):
		super(LinearAttention, self).__init__()
		self.heads = heads
		hidden_dim = dim_head * heads
		self.to_qkv = layers.Conv2D(hidden_dim * 3, 1, use_bias=False)
		self.to_out = layers.Conv2D(dim, 1)

		self.softmax = layers.Softmax(axis=-2)


	def call(self, x):
		print("In attention module")
		# b, c, h, w = x.shape # Original
		b, h, w, c = x.shape
		print(f"b {b}, h {h}, w {w}, c {c}")
		qkv = self.to_qkv(x)
		print(f"qkv {qkv}, shape {qkv.shape}")
		# q, k, v = rearrange(
		# 	qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
		# 	heads=self.heads, qkv=3
		# )
		# k = tf.nn.softmax(k, axis=-1)
		# context = tf.einsum('bhdn,bhen->bhde', k, v)
		# out = tf.einsum('bhde,bhdn->bhen', context, q)
		# out = rearrange(
		# 	out, 'b heads c (h w) -> b (heads c) h w', 
		# 	heads=self.heads, h=h, w=w
		# )

		# Alternative curtesy of ChatGPT
		q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)
		print(f"q {q}, shape {q.shape}")
		print(f"k {k}, shape {k.shape}")
		print(f"v {v}, shape {v.shape}")
		# q = tf.reshape(q, [b, h * w, self.heads, -1]) # Original from chatGPT
		# k = tf.reshape(k, [b, h * w, self.heads, -1]) # Original from chatGPT
		# v = tf.reshape(v, [b, h * w, self.heads, -1]) # Original from chatGPT
		q = tf.reshape(q, [b, -1, h * w, self.heads])
		k = tf.reshape(k, [b, -1, h * w, self.heads])
		v = tf.reshape(v, [b, -1, h * w, self.heads])
		print(f"q {q}, shape {q.shape}")
		print(f"k {k}, shape {k.shape}")
		print(f"v {v}, shape {v.shape}")
		# k = tfa.activations.sparsemax(k, axis=-1) # Original from chatGPT
		# k = tfa.activations.sparsemax(k, axis=-2) # Changed axis to be the same proper dim as in Pytorch
		# k = tf.nn.softmax(k, axis=-2) # Use softmax (with updated axis)
		k = self.softmax(k) # Use tf.keras.layer softmax (with updated axis)
		print(f"k softmax {k}, shape {k.shape}")
		# context = tf.einsum('bhdc,bhpc->bhdp', v, k) # Original from chatGPT. Seems to OOM on GPU at this step (even when batch size is 4). Creates a tensor with shape (B, 32, 13760, 13760) <- Probably what caused the OOM.
		context = tf.einsum('bdwc,bpwc->bdpc', v, k) # Creates a tensor with shape (B, 32, 32, 4)
		print(f"context {context}, shape {context.shape}")
		# out = tf.einsum('bhdp,bhdc->bhpc', context, q) # Original from chatGPT. Seems to OOM on GPU at this step (even when batch size is 4). Creates a tensor with shape (B, 32, 13760, 13760) <- Probably what caused the OOM.
		out = tf.einsum('bhec,bhnc->benc', context, q) # Creates a tensor with shape (B, 32, 13760, 4)
		print(f"out {out}, shape {out.shape}")
		# out = tf.reshape(out, [b, h, w, self.heads * -1]) Original from chatGPT. Invalid
		out = tf.reshape(out, [b, h, w, -1])
		print(f"out rearranged {out}, shape {out.shape}")
		temp = self.to_out(out)
		print(f"out to_out {temp}, shape {temp.shape}")
		return self.to_out(out)


class Residual(layers.Layer):
	def __init__(self, fn):
		super(Residual, self).__init__()
		self.fn = fn


	def call(self, x, *args, **kwargs):
		output = self.fn(x, *args, **kwargs) + x
		return output


class SinusoidalPosEmb(layers.Layer):
	def __init__(self, dim):
		super(SinusoidalPosEmb, self).__init__()
		self.dim = dim


	def call(self, x, scale=1000):
		half_dim = self.dim // 2
		emb = math.log(10000) / (half_dim - 1)
		emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
		emb = scale * tf.expand_dims(x, 1) * tf.expand_dims(emb, 0)
		emb = tf.concat((tf.math.sin(emb), tf.math.cos(emb)), axis=-1)
		return emb


class GradLogPEstimator2D(layers.Layer):
	def __init__(self, dim, dim_mults=(1, 2, 4), groups=8, 
			n_spkrs=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
		super(GradLogPEstimator2D, self).__init__()
		self.dim = dim
		self.dim_mults = dim_mults
		self.groups = groups
		self.n_spkrs = n_spkrs if not isinstance(n_spkrs, type(None)) else 1
		self.spk_emb_dim = spk_emb_dim
		self.pe_scale = pe_scale

		if n_spkrs > 1:
			self.spk_mlp = Sequential([
				layers.Dense(spk_emb_dim * 4),
				Mish(),
				layers.Dense(n_feats)
			])

		self.time_pos_emb = SinusoidalPosEmb(dim)
		self.mlp = Sequential([
			layers.Dense(dim * 4), 
			Mish(),
			layers.Dense(dim)
		])

		dims = [
			2 + (1 if n_spkrs > 1 else 0), 
			*map(lambda m: dim * m, dim_mults)
		]
		in_out = list(zip(dims[:-1], dims[1:]))
		self.downs = []
		self.ups = []
		num_resolutions = len(in_out)

		for ind, (dim_in, dim_out) in enumerate(in_out):
			is_last = ind >= (num_resolutions - 1)
			self.downs.append([
				ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
				ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
				Residual(ReZero(LinearAttention(dim_out))),
				DownSample(dim_out) if not is_last else Identity()
			])

		mid_dim = dims[-1]
		self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
		self.mid_attn = Residual(ReZero(LinearAttention(mid_dim)))
		self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

		for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
			self.ups.append([
				ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
				ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
				Residual(ReZero(LinearAttention(dim_in))),
				UpSample(dim_in)
			])

		self.final_block = Block(dim)
		self.final_conv = layers.Conv2D(1, 1)
	

	def call(self, x, mask, mu, t, spk=None):
		print(f"x {x}\nshape {x.shape}")
		print(f"mask {mask}\nshape {mask.shape}")
		print(f"mu {mu}\nshape {mu.shape}")
		print(f"t {t}\nshape {t.shape}")
		print(f"spk {spk}")
		if not isinstance(spk, type(None)):
			s = self.spk_mlp(spk)

		t = self.time_pos_emb(t, scale=self.pe_scale)
		print(f"t after time_pos_emb: {t}, shape {t.shape}")
		t = self.mlp(t)
		print(f"t after mlp: {t}, shape {t.shape}")

		if self.n_spkrs < 2:
			x = tf.stack([mu, x], axis=1)
			print("less than 2 n_spkr (1 n_spkr)")
			print(f"x after stack {x}, shape {x.shape}")
		else:
			s = tf.repeat(tf.expand_dims(s, -1), [1, 1, x.shape([-1])])
			x = tf.stack([mu, x, s], axis=1)
			print("more than 1 n_spkr")
			print(f"x after stack {x}, shape {x.shape}")
		mask = tf.expand_dims(mask, 1)
		print(f"mask unsqueezed {mask}, shape {mask.shape}")

		# Transpose so that correct dims are operated on.
		mask = tf.transpose(mask, [0, 2, 3, 1])
		x = tf.transpose(x, [0, 2, 3, 1])

		hiddens = []
		masks = [mask]
		for resnet1, resnet2, attn, downsample in self.downs:
			mask_down = masks[-1]
			x = resnet1(x, mask_down, t)
			x = resnet2(x, mask_down, t)
			x = attn(x)
			hiddens.append(x)
			x = downsample(x * mask_down)
			# print(f"mask_down {mask_down.shape}")
			# print(f"mask_down {mask_down[:, :, ::2, :].shape}")
			# masks.append(mask_down[:, :, :, ::2]) # Original
			masks.append(mask_down[:, :, ::2, :])

		masks = masks[:-1]
		mask_mid = masks[-1]
		x = self.mid_block1(x, mask_mid, t)
		x = self.mid_attn(x)
		x = self.mid_block2(x, mask_mid, t)

		for resnet1, resnet2, attn, upsample in self.ups:
			mask_up = masks.pop()
			# print(f"x {x.shape}")
			# x = tf.concat((x, hiddens.pop()), axis=1) # Original
			x = tf.concat((x, hiddens.pop()), axis=-1)
			# print(f"x concat {x.shape}")
			x = resnet1(x, mask_up, t)
			x = resnet2(x, mask_up, t)
			x = attn(x)
			x = upsample(x * mask_up)

		x = self.final_block(x, mask)
		print(f"x final block {x}, shape {x.shape}")
		output = self.final_conv(x * mask)
		print(f"output final conv {output}, shape {output.shape}")
		print(f"mask shape {mask.shape}")
		temp = tf.squeeze(output * mask, -1)
		print(f"output * mask squeezed shape {temp.shape}")

		# return tf.squeeze(output * mask, 1) # Original
		return  tf.squeeze(output * mask, -1)


def get_noise(t, beta_init, beta_term, cumulative=False):
	if cumulative:
		noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t ** 2)
	else:
		noise = beta_init + (beta_term - beta_init) * t
	return noise


class Diffusion(keras.Model):
	def __init__(self, n_mel_channels, dim, n_spkrs=1, spk_emb_dim=64,
			beta_min=0.05, beta_max=20, pe_scale=1000):
		super(Diffusion, self).__init__()
		self.n_mel_channels = n_mel_channels
		self.dim = dim
		self.n_spkrs = n_spkrs
		self.spk_emb_dim = spk_emb_dim
		self.beta_min = beta_min
		self.beta_max = beta_max
		self.pe_scale = pe_scale

		self.estimator = GradLogPEstimator2D(
			dim, n_spkrs=n_spkrs, spk_emb_dim=spk_emb_dim, 
			pe_scale=pe_scale
		)


	def forward_diffusion(self, x0, mask, mu, t):
		print(f"x0 {x0}, shape {x0.shape}")
		print(f"mask {mask}, shape {mask.shape}")
		print(f"mu {mu}, shape {mu.shape}")
		print(f"t {t}, shape {t.shape}")
		time = tf.expand_dims(tf.expand_dims(t, -1), -1)
		print(f"time {time}, shape {time.shape}")
		cum_noise = get_noise(
			time, self.beta_min, self.beta_max, cumulative=True
		)
		print(f"cum_noise {cum_noise}, shape {cum_noise.shape}")
		mean = x0 * tf.math.exp(-0.5 * cum_noise) +\
			mu * (1.0 - tf.math.exp(-0.5 * cum_noise))
		print(f"mean {mean}, shape {mean.shape}")
		variance = 1.0 - tf.math.exp(-cum_noise)
		print(f"variance {variance}, shape {variance.shape}")
		z = tf.random.normal(x0.shape, dtype=x0.dtype)
		print(f"z {z}, shape {z.shape}")
		xt = mean + z * tf.math.sqrt(variance)
		print(f"xt {xt}, shape {xt.shape}")
		return xt * mask, z * mask


	def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
		h = 1.0 / n_timesteps
		xt = z * mask
		for i in range(n_timesteps):
			t = (1.0 - (i + 0.5) * h) * tf.ones(
				z.shape[0], dtype=z.dtype
			)
			time = tf.expand_dims(tf.expand_dims(t, -1), -1)
			noise_t = get_noise(
				time, self.beta_min, self.beta_max, cumulative=False
			)
			if stoc: # Adds stochastic term
				dxt_det = 0.5 * (mu - xt) - self.estimator(
					xt, mask, mu, t, spk
				)
				dxt_det = dxt_det * noise_t * h
				dxt_stoc = tf.random.normal(z.shape, dtype=z.dtype)
				dxt_stoc = dxt_stoc * tf.math.sqrt(noise_t * h)
				dxt = dxt_det + det_stoc
			else:
				dxt = 0.5 * (
					mu - xt - self.estimator(xt, mask, mu, t, spk)
				)
				dxt = dxt * noise_t * h
			xt = (xt - dxt) * mask
		return xt


	def call(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
		return self.reverse_diffusion(
			z, mask, mu, n_timesteps, stoc, spk
		)


	def loss_t(self, x0, mask, mu, t, spk=None):
		xt, z = self.forward_diffusion(x0, mask, mu, t)
		time = tf.expand_dims(tf.expand_dims(t, -1), -1)
		cum_noise = get_noise(
			time, self.beta_min, self.beta_max, cumulative=True
		)
		noise_estimation = self.estimator(xt, mask, mu, t, spk)
		noise_estimation *= tf.math.sqrt(1.0 - tf.math.exp(-cum_noise))
		loss = tf.math.reduce_sum((noise_estimation + z) ** 2) /\
			(tf.math.reduce_sum(mask) * self.n_mel_channels)
		return loss, xt


	def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
		print(f"x0 {x0}, shape {x0.shape}")
		t = tf.random.uniform(shape=(x0.shape[0],), dtype=x0.dtype)
		t = tf.clip_by_value(t, offset, 1.0 - offset)
		return self.loss_t(x0, mask, mu, t, spk)