import argparse
import logging
import os
import time
from datetime import datetime, timedelta

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-size", type=int, default=100, help="message size in MiB")
    parser.add_argument("--duration", type=float, default=10, help="traffic duration in seconds")
    parser.add_argument(
        "--interval", type=float, default=0,
        help="seconds to wait after each transfer (reduces GPU DMA duty cycle)",
    )
    parser.add_argument("--logdir", type=str, default="/workspace/Greyhound/trainlog")
    parser.add_argument(
        "--device", type=int, default=None,
        help="local CUDA device (default: LOCAL_RANK, or GPU 0 when unset)",
    )
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def configure_logging(logdir, rank):
    os.makedirs(logdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    logpath = os.path.join(logdir, f"comm_worker_rank{rank}_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logpath), logging.StreamHandler()],
    )
    return logpath


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or rank not in (0, 1):
        raise ValueError("single_comm.py requires WORLD_SIZE=2 and RANK=0 or RANK=1")
    if args.tensor_size <= 0 or args.duration <= 0 or args.interval < 0:
        raise ValueError("size/duration must be positive and interval must be non-negative")

    device_index = args.device
    if device_index is None:
        device_index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    logpath = configure_logging(args.logdir, rank)

    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=args.timeout),
    )
    element_size = torch.empty((), dtype=torch.float32).element_size()
    numel = args.tensor_size * 1024 * 1024 // element_size
    tensor = torch.randn(numel, device=device) if rank == 0 else torch.empty(numel, device=device)

    dist.all_reduce(torch.ones(1, device=device))
    dist.barrier()
    start = time.monotonic()
    transfers = 0
    control = torch.ones(1, dtype=torch.int32, device=device)
    while True:
        if rank == 0:
            control.fill_(int(time.monotonic() - start < args.duration))
            dist.send(control, dst=1)
            if not control.item():
                break
            dist.send(tensor, dst=1)
        else:
            dist.recv(control, src=0)
            if not control.item():
                break
            dist.recv(tensor, src=0)
        transfers += 1
        if args.interval:
            time.sleep(args.interval)
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.monotonic() - start
    bandwidth = args.tensor_size * transfers / elapsed
    logging.info(
        "rank=%d device=%d transfers=%d tensor_size=%d MiB elapsed=%.3f s bandwidth=%.2f MiB/s log=%s",
        rank, device_index, transfers, args.tensor_size, elapsed, bandwidth, logpath,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
