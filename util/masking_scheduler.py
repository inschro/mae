

class MaskingScheduler:
    def __init__(self, initial_ratio=0.3, final_ratio=0.9, warmup_epochs=50,steps_per_epoch = 512):
        if initial_ratio > 0.006:
            self.initial_ratio = initial_ratio
        else:
            print("Warning: SETTING initial_ratio TO 0.006. When using maskingScheduler with the vanilla Vit family the smallest  \
                  possible ratio is 0.006 as atleast one patch (1:196) has to be masked. ")
            self.initial_ratio = 0.006
        
        self.final_ratio = final_ratio
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch

    def __call__(self, step):
        if step < (self.warmup_epochs*self.steps_per_epoch):
            return self._warmup_schedule(step)
        else:
            return self.final_ratio

    def _warmup_schedule(self, step):
        ratio = self.initial_ratio + (self.final_ratio - self.initial_ratio) * (step / (self.warmup_epochs*self.steps_per_epoch))
        return ratio

    #def _main_schedule(self, step):
    #    progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
    #    ratio = self.max_ratio - (self.max_ratio - self.final_ratio) * progress
    #    return self._ensure_minimum_masking(ratio)
