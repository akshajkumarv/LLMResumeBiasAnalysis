import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="data", help=f"Path to the dataset. Default: data")

parser.add_argument("--model_name", type=str, default="bard", help="Name of the model to run. Can be gpt, bard, or claude")

parser.add_argument("--email_domain", type=str, default="yahoo", help="You can specify an email domain")

parser.add_argument("--political_orientation", action=argparse.BooleanOptionalAction, type=bool, default=False, help="Political orientation. Whether to include political orientation")
parser.add_argument("--pregnancy", action=argparse.BooleanOptionalAction, type=bool, default=False, help="Pregnancy. Whether to include pregnancy")
parser.add_argument("--employment_gap", action=argparse.BooleanOptionalAction, type=bool, default=False, help="Employment Gap. Whether to include employment gap")

parser.add_argument('--temperature', type=float, default=0.0) 

parser.add_argument('--api_key_file', type=str, required=False, help='Name of the API key file.')

parser.add_argument('--cid_lambda', type=float, default=50.0, help='Lambda for CID')

# add argument for llama model, default to 7b, can only be set if model_name is llama
parser.add_argument('--llama_model', type=str, choices=['7b-chat', '13b-chat'], help='Name of the llama model to run.')

args = parser.parse_args()

# check that only one of political orientation, pregnancy, or employment gap is true
if sum([args.political_orientation, args.pregnancy, args.employment_gap]) > 1:
    raise ValueError("Only one of political orientation, pregnancy, or employment gap can be true")

# check that llama model is only set if model_name is llama
if args.model_name != 'llama' and args.llama_model is not None:
    raise ValueError("llama_model can only be set if model_name is llama")

# check that llama model is set if model_name is llama
if args.model_name == 'llama' and args.llama_model is None:
    raise ValueError("llama_model must be set if model_name is llama")