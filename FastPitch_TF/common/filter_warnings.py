# filter_warnings.py


import warnings


# NGC 22.04-py3 container (PyTorch 1.12.0a0+bd13bc6)
warnings.filterwarnings(
	"ignore",
	message='positional arguments and argument "destination" are deprecated.'
			' nn.Module.state_dict will not accept them in the future.'
)

# 22.08-py3 container
warnings.filterwarnings(
	"ignore",
	message="is_namedtuple is deprecated, please use the python checks"
)
