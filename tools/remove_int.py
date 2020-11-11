import torch

def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="convert pycls weights, outputs the model with '-d2' ")
    parser.add_argument(
        "--weights",
        required=True,
        help="weight file for model weights",
    )


if __name__ == "__main__":
    args = parse_args()
    in_path = args.paths
    ckpt = torch.load(in_path)
    model = {'model': ckpt['model_state'], 'matching_heuristics': True}

    out_path = in_path.split('.')[0] + '-d2.pth'

    torch.save(model, out_path)
    