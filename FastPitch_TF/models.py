# models.py


from common.text.symbols import get_symbols, get_pad_idx
from common.utils import DefaultAttrDict


def parse_model_args(model_name, parser, add_help=False):
	if model_name == "FastPitch":
		import arg_parser
		return arg_parser.parse_fastpitch_args(parser, add_help)


def get_fastpitch_config(args):
	# Mark keys missing in `args` with an object (None is ambiguous)
	_missing = object()
	args = DefaultAttrDict(lambda: _missing, vars(args))

	model_config = dict(
		# io
		n_mel_channels=args.n_mel_channels,
		# symbols
		n_symbols=(len(get_symbols(args.symbol_set))
					if args.symbol_set is not _missing else _missing),
		padding_idx=(get_pad_idx(args.symbol_set)
					if args.symbol_set is not _missing else _missing),
		symbols_embedding_dim=args.symbols_embedding_dim,
		# input FFT
		in_fft_n_layers=args.in_fft_n_layers,
		in_fft_n_heads=args.in_fft_n_heads,
		in_fft_d_head=args.in_fft_d_head,
		in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
		in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
		in_fft_output_size=args.in_fft_output_size,
		p_in_fft_dropout=args.p_in_fft_dropout,
		p_in_fft_dropatt=args.p_in_fft_dropatt,
		p_in_fft_dropemb=args.p_in_fft_dropemb,
		# output FFT
		out_fft_n_layers=args.out_fft_n_layers,
		out_fft_n_heads=args.out_fft_n_heads,
		out_fft_d_head=args.out_fft_d_head,
		out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
		out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
		out_fft_output_size=args.out_fft_output_size,
		p_out_fft_dropout=args.p_out_fft_dropout,
		p_out_fft_dropatt=args.p_out_fft_dropatt,
		p_out_fft_dropemb=args.p_out_fft_dropemb,
		# duration predictor
		dur_predictor_kernel_size=args.dur_predictor_kernel_size,
		dur_predictor_filter_size=args.dur_predictor_filter_size,
		p_dur_predictor_dropout=args.p_dur_predictor_dropout,
		dur_predictor_n_layers=args.dur_predictor_n_layers,
		# pitch predictor
		pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
		pitch_predictor_filter_size=args.pitch_predictor_filter_size,
		p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
		pitch_predictor_n_layers=args.pitch_predictor_n_layers,
		# pitch conditioning
		pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
		# speakers parameters
		n_speakers=args.n_speakers,
		speaker_emb_weight=args.speaker_emb_weight,
		# energy predictor
		energy_predictor_kernel_size=args.energy_predictor_kernel_size,
		energy_predictor_filter_size=args.energy_predictor_filter_size,
		p_energy_predictor_dropout=args.p_energy_predictor_dropout,
		energy_predictor_n_layers=args.energy_predictor_n_layers,
		# energy conditioning
		energy_conditioning=args.energy_conditioning,
		energy_embedding_kernel_size=args.energy_embedding_kernel_size,
	)

	# Fill missing keys from model_config.
	final_config = {}
	missing_keys = set(model_config.keys()) - set(final_config.keys())
	final_config.update({k: model_config[k] for k in missing_keys})

	assert all(v is not _missing for v in final_config.values())
	return final_config