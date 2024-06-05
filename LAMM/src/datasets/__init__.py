import torch
from torch.utils.data import DataLoader
from .samplers import DistributedBatchSampler
from .dataset import LAMMDataset, OctaviusDataset


def collate_fn(batch):
    res = dict()
    keys = batch[0].keys()
    for key in keys:
        res[key] = [data[key] for data in batch]
    return res


def load_data(data_path):

    dataset = OctaviusDataset(
        data_path
    )
    # dataset = OctaviusDataset(
    #     args["data_path_2d"], args["data_path_3d"], args["vision_root_path_2d"],
    #     args["vision_root_path_3d"], args["loop_2d"], args["loop_3d"],
    # )

    
    return dataset


def load_dataset(args, data_path):
    """load LAMM datasets

    :param dict args: input arguments
    :return tupe: dataset, dataloader, sampler
    """
    dataset = load_data(data_path)
    sampler = torch.utils.data.RandomSampler(dataset)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = (
        args["world_size"] * args["dschf"].config["train_micro_batch_size_per_gpu"]
    )
    batch_sampler = DistributedBatchSampler(sampler, batch_size, True, rank, world_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=1,
        collate_fn=dataset.collate,
        pin_memory=True,
    )
    return dataset, dataloader, sampler
