import abc
import torch
import torch.nn.functional as F
from SEDD.graph_lib import Graph
from SEDD.noise_lib import Noise
from catsample import sample_categorical

from model import utils as mutils
import sys
import json


class PrintSamples(object):
    def __init__(self, tokenizer, stream=None, json_file=None):
        self.tokenizer = tokenizer
        self.stream = stream or sys.stdout
        if json_file:
            self.json_file = open(json_file, "w")
        else:
            self.json_file = None

    def decode(self, sample):
        return self.tokenizer.batch_decode(sample)

    def print(self, samples, **kwargs):
        text_samples = self.decode(samples)
        assert len(text_samples) == 1
        text_tokens = self.tokenizer.convert_ids_to_tokens(
            samples.squeeze(0).tolist()
        )
        # the text_tokens can have encoded space characters. Replace with space for better readability.
        text_tokens = [i.replace("Ġ", " ") for i in text_tokens]
        for i in text_samples:
            # print on the stream
            print(kwargs, file=self.stream)
            print(i, file=self.stream)
            print(
                "=================================================",
                file=self.stream,
            )
        if self.json_file:
            self.to_json(samples, text_tokens, **kwargs)

    def to_json(self, samples, text_samples, **kwargs):
        json_str = json.dumps({"text": text_samples, **kwargs})
        self.json_file.write(json_str)
        self.json_file.write("\n")


_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f"Already registered model with name: {local_name}"
            )
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise, printer=None):
        super().__init__()
        self.graph = graph
        self.noise = noise
        self.printer = printer or (lambda *args, **kwargs: None)

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = (
            step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        )
        x = self.graph.sample_rate(x, rev_rate)
        return x


@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        # breakpoint()
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(
            x, curr_sigma
        )  # shape: (batch_size, seq_len, vocab_size) (vocab includes mask)
        # note that the score_fn internally sets log_score for mask to be 0
        # so the score = exp(log_score) is 1 for mask.
        # The score for other classes will be between 0 and 1.

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        sample = sample_categorical(probs)
        # breakpoint()
        self.printer.print(
            sample, step=t.detach().item(), level="intermediate"
        )
        return sample


class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        # breakpoint()
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        # return probs.argmax(dim=-1)
        return sample_categorical(probs)


def get_sampling_fn(config, graph, noise, batch_dims, eps, device):

    sampling_fn = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims,
        predictor=config.sampling.predictor,
        steps=config.sampling.steps,
        denoise=config.sampling.noise_removal,
        eps=eps,
        device=device,
    )

    return sampling_fn


def get_pc_sampler(
    graph: Graph,
    noise: Noise,
    batch_dims,
    predictor,
    steps,
    denoise=True,
    eps=1e-5,
    device=torch.device("cpu"),
    proj_fun=lambda x: x,
    printer=None,
):
    """
    Returns a function that performs probabilistic counting (PC) sampling.

    Args:
        graph (Graph): The graph object.
        noise (Noise): The noise object.
        batch_dims: The dimensions of the batch (batch_size, seq_len).
        predictor: The predictor function.
        steps (int): The number of denoising steps for sampling.
        denoise (bool, optional): Whether to perform denoising step. Defaults to True.
        eps (float, optional): The value of epsilon. Defaults to 1e-5.
        device (torch.device, optional): The device to use. Defaults to torch.device('cpu').
        proj_fun (function, optional): The projection function. Defaults to lambda x: x.  Does nothing right now.

    Returns:
        function: The PC sampler function.
    """
    predictor = get_predictor(predictor)(graph, noise, printer=printer)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(
            model, train=False, sampling=True
        )
        x = graph.sample_limit(*batch_dims).to(
            device
        )  # start with a random sample
        timesteps = torch.linspace(
            1, eps, steps + 1, device=device
        )  # timesteps for denoising
        dt = (1 - eps) / steps  # delta

        for i in range(steps):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=device
            )  # current timestep of shape (batch_size, 1)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)

        return x

    return pc_sampler
