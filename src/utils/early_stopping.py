class EarlyStopper:
    def __init__(self, PATIENCE=1, MIN_DELTA=0):
        self.counter = 0
        self.min_delta = MIN_DELTA
        self.patience = PATIENCE  # Num of epoch that val loss is allowed to increase
        self.min_val_loss = float('inf')

    def check(self, val_loss: float):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
