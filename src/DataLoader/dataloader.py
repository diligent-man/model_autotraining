from .DefaultDataLoader import DefaultDataLoader

__all = ["available_dataloader"]

available_dataloader = {
    "DefaultDataLoader": DefaultDataLoader
}
