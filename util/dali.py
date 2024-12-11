# dali.py
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import nvidia.dali.types as dalitypes

# Modify the DALI pipeline to only read and decode images
@pipeline_def
def dali_train_pipeline(data_path, name="Reader", random_shuffle=True, initial_fill=4096, size=224, shard_id=0, num_shards=1):
    jpegs, labels = fn.readers.file(
        name=name,
        file_root=data_path,
        random_shuffle=random_shuffle, 
        initial_fill=initial_fill,
        shard_id=shard_id,
        num_shards=num_shards,
    )

    images = fn.decoders.image_random_crop(
        jpegs,
        device='mixed',
        random_area=[0.25, 1.0],
    )

    images = fn.resize(
        images,
        size=size,
        interp_type=dalitypes.DALIInterpType.INTERP_CUBIC
    )

    flip = fn.random.coin_flip(probability=0.5)
    images = fn.flip(images, horizontal=flip, vertical=0)
    
    return images, labels

# Create a custom DaliDataloader class
class DaliDataloader:
    def __init__(self, data_path, batch_size, num_threads=4, transforms=None, name="Reader", num_gpus=1, input_size=224, device_id=0):
        print(f"Creating DALI dataloader for {num_gpus} with shard_id {device_id}")
        # Create the DALI pipeline
        self.pipeline = dali_train_pipeline(
            batch_size=batch_size,
            size=input_size,
            name=name,
            shard_id=device_id,
            num_shards=num_gpus,
            num_threads=num_threads,
            device_id=device_id,
            data_path=data_path,
            prefetch_queue_depth={"cpu_size": 2, "gpu_size": 2},
        )
        self.pipeline.build()

        # Create the DALI iterator
        self.dali_iterator = DALIClassificationIterator(
            pipelines=self.pipeline,
            reader_name=name,
            auto_reset=False,
        )
        self.transforms = transforms

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.dali_iterator)

    def __next__(self):
        try:
            data = next(self.dali_iterator)
        except StopIteration:
            self.dali_iterator.reset()
            raise StopIteration
        images = data[0]['data']
        labels = data[0]['label'].squeeze().long()

        # Convert images from [batch_size, H, W, C] to [batch_size, C, H, W]
        images = images.permute(0, 3, 1, 2)
        # Convert images to float and normalize to [0, 1]
        images = images.float() / 255.0

        # Ensure labels are on the same device as images
        labels = labels.to(images.device)

        # Apply transforms
        if self.transforms:
            images = self.transforms(images)
        return images, labels


if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataloader = DaliDataloader(
        data_path="/beegfs/data/shared/imagenet/imagenet100/train",
        batch_size=16,
        num_threads=2,
        device_id=0,
        transforms=transform,
    )


    for batch in dataloader:
        imgs, labels = batch
        print(f"Images shape: {imgs.shape}, Labels shape: {labels.shape}")
        print(f"Imgs dtype: {imgs.dtype}, Labels dtype: {labels.dtype}")
        print(f"imgs.device: {imgs.device}, labels.device: {labels.device}")
        print("----------------")

        imgs_min, imgs_max = imgs.min(), imgs.max()
        labels_min, labels_max = labels.min(), labels.max()
        print(f"Images min: {imgs_min}, Images max: {imgs_max}")
        print(f"Labels min: {labels_min}, Labels max: {labels_max}")
        print("----------------")

        print(f"First image: {imgs[0]}, \n First label: {labels[0]}")
        break