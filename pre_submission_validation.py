# imports
import torch
import yaml
import time
from model import Model
from utils.chess_gameplay import Agent, play_game

# model instantiation
model_config = yaml.safe_load(open("model_config.yaml"))
model0 = Model(**model_config)

# checkpoint loading
checkpoint = torch.load("checkpoint.pt", map_location=torch.device('cpu'))
model0.load_state_dict(checkpoint["model"])

# model inference
pgn = "1.d4 Nf6 2.c4 d5 3.Nf3 e6 4.Nc3 Nc6 5."
move = "e3"
score = model0.score(pgn, move)

# outputs validation
assert isinstance(score, float), "ERROR: Model score method must return a float."
print("Outputs pass validation tests.")

# testing gameplay
model1 = Model(**model_config)
agent0, agent1 = Agent(model0), Agent(model1)
gameplay_kwargs = {
    "agents": {'white': agent0, 'black': agent1},
    "teams": {"white": "Adam", "black": "Ben"},
    "max_moves": 20,
    "min_seconds_per_move": 0.0,
    "verbose": False,
    "poseval": False,
    "image_path": None
}

timer_start = time.perf_counter()
game_result = play_game(**gameplay_kwargs)
elapsed = time.perf_counter() - timer_start
assert elapsed < 160, "Model too slow, consider simplifying or reducing the size of your model."
print("Model passes validation test.")