import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument(
        "--model_path", default="louaaron/sedd-medium", type=str
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # add the mask token at the end of the vocabulary
    tokenizer.add_special_tokens({"mask_token": "<M>"})
    if args.log_file:
        printer = sampling.PrintSamples(tokenizer, None, args.log_file)
    else:
        printer = sampling.PrintSamples(tokenizer)
    # breakpoint()
    sampling_fn = sampling.get_pc_sampler(
        graph,
        noise,
        (args.batch_size, 1024),
        "analytic",
        args.steps,
        device=device,
        printer=printer,
    )

    samples = sampling_fn(model)
    printer.print(samples, step=args.steps, level="final")


if __name__ == "__main__":
    main()
