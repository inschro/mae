import torch
import time
from torch.profiler import profile, ProfilerActivity

class Profiler:
    def __init__(self, args=None, logger=None) -> None:
        self.args = args or {}
        self.profiler = None
        self.logger = logger # Tensorboard SummaryWriter
        self.current_epoch = None
        self.active = self.args.get("use_profiler", False)

        self.config = {
            "activities": [ProfilerActivity.CPU],
            "with_flops": True,
            "active_epochs": range(0, 100, 5),
        }

    def configure(self, config, erase=False):
        if erase:
            self.config = config
        else:
            self.config.update(config)

    def __call__(self, current_epoch=None):
        self.current_epoch = current_epoch
        return self

    def __enter__(self):
        if not self.active:
            return
        if not self.current_epoch in self.config["active_epochs"]:
            return
        if self.current_epoch is not None and self.config.get("active_epochs", None) is None:
            raise ValueError("Please provide active_epochs in the config")
        if self.profiler is not None:
            raise ValueError("Profiler already active")
        if self.config.get("activities", None) is None:
            raise ValueError("Please provide activities in the config")
        
        self.profiler = profile(
            activities=self.config["activities"],
            with_flops=self.config.get("with_flops", False),
        )
        self.profiler.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self._handle_exception(exc_type, exc_value, traceback)
        if self.profiler is None:
            return
        self._log_when_exit()
        del self.profiler
        self.profiler = None
        

    def _log_when_exit(self):
        if self.profiler is None:
            return
        if self.current_epoch is not None:
            epoch = self.current_epoch
            print(f"Stats for epoch {self.current_epoch}")
        else:
            epoch = 0
            print("Stats for this epoch")
        self.profiler.__exit__(None, None, None)
        key_averages = self.profiler.key_averages()
        if self.config.get("with_flops", False):
            flops = sum([item.flops for item in key_averages])
            print(f"FLOPS: {flops}")
            self.logger.add_scalar("FLOPS", flops, epoch)
        

    def _handle_exception(self, exc_type, exc_value, traceback):
        print("Exception occurred, stopping profiler")
        self._log_when_exit()
        return False




# target usage:
if __name__ == "__main__":
    def train_epoch():
        size = torch.randint(100, 1000, (1,)).item()
        for _ in range(torch.randint(1, 10, (1,)).item()):
            tensor1 = torch.randn(size, size)
            tensor2 = torch.randn(size, size)
            result = tensor1 @ tensor2

    args = {"use_profiler": True}

    profiler = Profiler(args)
    for epoch in range(20):
        with profiler(current_epoch=epoch):
            train_epoch()
        # exiting the context will log the compute stats