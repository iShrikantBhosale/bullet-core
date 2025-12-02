class LossMonitor:
    def __init__(self, factor=2.0, history=20):
        self.factor = factor
        self.history = history
        self.losses = []

    def update(self, loss):
        self.losses.append(loss)
        if len(self.losses) > self.history:
            self.losses.pop(0)

    def spike(self, loss):
        if len(self.losses) < 5: return False
        avg = sum(self.losses) / len(self.losses)
        return loss > avg * self.factor

    def recover_message(self, loss, avg):
        return (
            f"⚠️ LOSS SPIKE: {loss:.4f} > {avg:.4f} x{self.factor} "
            f"→ Lowering LR + restoring last good weights"
        )
