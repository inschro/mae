# dali.py
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import nvidia.dali.fn as fn

@pipeline_def
def dali_train_pipeline(data_path, input_size):
    jpegs, labels = fn.readers.file(name="Reader", file_root=data_path, random_shuffle=True, initial_fill=4096)
    jpegs.gpu()
    images = fn.decoders.image(jpegs, device='mixed')
    images = fn.resize(images, resize_x=input_size, resize_y=input_size)
    images = fn.crop_mirror_normalize(
        images,
        crop=(input_size, input_size),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip()
    )
    return images, labels

def get_dali_dataloader(data_path, batch_size, input_size, num_threads=4, device_id=0):
    # Create and build the DALI pipeline
    pipeline = dali_train_pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        data_path=data_path,
        input_size=input_size,
        prefetch_queue_depth={"cpu_size": 2, "gpu_size": 4},
    )
    pipeline.build()
    
    # Create the DALI iterator
    dali_iterator =  DALIClassificationIterator(
        pipelines=pipeline,
        reader_name="Reader",
        auto_reset=False,
    )
    
    return dali_iterator

@pipeline_def
def dali_linprobe_pipeline(data_path, input_size, shard_id=0, num_shards=1):
    jpegs, labels = fn.readers.file(
        name="reader_linprobe",
        file_root=data_path,
        random_shuffle=True,
        initial_fill=1024,
        shard_id=shard_id,
        num_shards=num_shards
    )
    jpegs.gpu()
    images = fn.decoders.image(jpegs, device='mixed')
    images = fn.resize(images, resize_x=input_size, resize_y=input_size)
    images = fn.crop_mirror_normalize(
        images,
        crop=(input_size, input_size),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip()
    )
    return images, labels

def get_dali_dataloader_linprobe_train(data_path, batch_size, input_size, num_threads=4, num_gpus=1):
    # Create a pipeline for each GPU and store them in a list
    pipelines = []
    for device_id in range(num_gpus):
        pipeline = dali_linprobe_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            data_path=data_path,
            input_size=input_size,
            shard_id=device_id,
            num_shards=num_gpus,
            prefetch_queue_depth={"cpu_size": 2, "gpu_size": 2},
        )
        pipeline.build()
        pipelines.append(pipeline)
    
    # Create a DALI iterator with all pipelines
    dali_iterator = DALIClassificationIterator(
        pipelines=pipelines,
        reader_name="reader_linprobe",
        auto_reset=False,
    )
    
    return dali_iterator

def get_dali_dataloader_linprobe_val(data_path, batch_size, input_size, num_threads=4, num_gpus=1):
    # Create a pipeline for each GPU and store them in a list
    pipelines = []
    for device_id in range(num_gpus):
        pipeline = dali_linprobe_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            data_path=data_path,
            input_size=input_size,
            shard_id=device_id,
            num_shards=num_gpus,
            prefetch_queue_depth={"cpu_size": 2, "gpu_size": 2},
        )
        pipeline.build()
        pipelines.append(pipeline)
    
    # Create a DALI iterator with all pipelines
    dali_iterator = DALIClassificationIterator(
        pipelines=pipelines,
        reader_name="reader_linprobe",
        auto_reset=False,
    )
    
    return dali_iterator

if __name__ == "__main__":
    
    dataloader = get_dali_dataloader(
        data_path="/media/ingo/datasets/imagenet-mini/val",
        batch_size=16,
        input_size=224,
        num_threads=2,
        device_id=0
    )

    for batch in dataloader:
        img, label = batch[0]["data"], batch[0]["label"]
        print(img.shape, label)
        break