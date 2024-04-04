__all__ = ['EarlyStopper']


class EarlyStopper:
    def __init__(self, min_val_loss: float = float("inf"), patience=1, min_delta=0):
        self.counter = 0
        self.min_delta = min_delta
        self.patience = patience  # Num of epoch that val loss is allowed to increase
        self.min_val_loss = min_val_loss

    def check(self, val_loss: float):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
