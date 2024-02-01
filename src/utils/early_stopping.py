class EarlyStopper:
    def __init__(self, PATIENCE=1, MIN_DELTA=0):
        self.counter = 0
        self.min_delta = MIN_DELTA
        self.patience = PATIENCE  # Num of epoch that val loss is allowed to increase
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
