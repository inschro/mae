## next steps for the project

### Training method on GPU cluster
1. [x] Fix PSNR calculation
1. [X] Fix NaN problem
3. [X] Test epoch on 3080 to estimate training time
3. [X] Find dataset splits
4. [X] Compromise on compute resources, dataset size, model size, number of epochs
5. [ ] Train baseline & our method on single GPU (maybe concurrently)
6. [X] Profiling of training time (theoretical/empirical)
7. Plot performance metrics
    1. [ ] ACC/FLOP
    2. [ ] ACC/Epoch
    3. [ ] PSNR while pretraining

### Other considerations
1. Masking methods
    1. [ ] Reverse entropy
    2. [ ] Image codec
    3. [ ] Binning
    4. [ ] Entropy loss scaling
2. [ ] Data augmentation
3. [ ] Parameter comparison of different models
4. [ ] Masking scheduler

Data path: 
/beegfs/data/shared/imagenet/ILSVRC/Data/CLS-LOC/

