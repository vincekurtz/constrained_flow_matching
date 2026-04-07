from datasets.star import StarDataset
from architectures.flow import FlowMLP
from examples.example_base import FlowExample
from flax import nnx

# Parse command line arguments (use --help to see options)
parser = FlowExample.build_arg_parser("data/star_model.pkl")
args = parser.parse_args()

# Define the architecture of the flow model we'll train.
model = FlowMLP(
    data_shape=(2,),
    time_embedding_size=4,
    hidden_sizes=(64, 64, 64, 64),
    rngs=nnx.Rngs(0),
)

# Define training hyperparameters
hyperparams = {
    "num_epochs": 5000,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "seed": 0,
    "print_frequency": 100,
}

# Run training and/or generation, depending on CLI flags
example = FlowExample(
    dataset=StarDataset(num_samples=1024),
    model=model,
    save_path=args.save_path,
    plot_lims=(-2, 2),
)
example.run(
    args,
    parser=parser,
    **hyperparams
)
