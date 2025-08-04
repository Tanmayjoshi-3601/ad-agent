from dataclass import dataclass

@dataclass
class Arm:
    key:str
    alpha: float = 1.0  # prior (and running) successes
    beta: float = 1.0 # prior (and running) failures
