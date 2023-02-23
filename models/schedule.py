
class LinearSchedule():
    """Linearly scaled scheduler."""

    def __init__(
        self,initial_value: float, final_value: float, num_steps: int,  start = 0, 
    ):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps
        self.start = start
    def get(self, step: int):
        """Get the value for the given step."""
        if step>self.start:
            if self.num_steps == 0:
                return self.final_value
            alpha = min(step / self.num_steps, 1.0)
            return (1.0 - alpha) * self.initial_value + alpha * self.final_value
        else:
            return self.initial_value 
    
class ExponentialSchedule():
    """Exponentially decaying scheduler."""

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        num_steps: int,
        eps: float = 1e-10,
    ):
        super().__init__()
        if initial_value <= final_value:
            raise ValueError("Final value must be less than initial value.")

        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps
        self.eps = eps

    def get(self, step: int):
        """Get the value for the given step."""
        if step >= self.num_steps:
            return self.final_value

        final_value = max(self.final_value, self.eps)
        base = final_value / self.initial_value
        exponent = step / (self.num_steps - 1)
        return self.initial_value * base**exponent
