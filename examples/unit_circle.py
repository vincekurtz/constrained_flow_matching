from datasets.unit_circle import UnitCircleDataset
from architectures.flow import FlowMLP
from examples.example_base import FlowExample
from flax import nnx

# Parse command line arguments (use --help to see options)
parser = FlowExample.build_arg_parser("data/unit_circle_model.pkl")
args = parser.parse_args()

# Define the architecture of the flow model we'll train.
model = FlowMLP(
    data_size=2,
    time_embedding_size=4,
    hidden_sizes=(64, 64),
    rngs=nnx.Rngs(0),
)

# Define training hyperparameters
hyperparams = {
    "num_epochs": 500,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "seed": 0,
    "print_frequency": 10,
}

# Run training and/or generation, depending on CLI flags
example = FlowExample(
    dataset=UnitCircleDataset(num_samples=1024),
    model=model,
    save_path=args.save_path,
    plot_lims=(-2, 2),
)
example.run(
    args,
    parser=parser,
    **hyperparams
)
