import argparse
import json
import os
import torch
from typing import Callable, List
import triton
from tqdm import tqdm
from jee_ops.bgmv_expand import bgmv_expand
from jee_ops.bgmv_shrink import bgmv_shrink

SAVE_DIR = os.path.join(os.path.dirname(__file__), "bgmv_config")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

HIDDEN_SIZE = [
    128,
    256,
    512,
    1024,
    1152,
    1280,
    1536,
    2048,
    2304,
    2560,
    2752,
    3072,
    3328,
    3456,
    3584,
    4096,
    4608,
    5120,
    5504,
    5632,
    6144,
    6400,
    6848,
    6912,
    7168,
    8192,
    9216,
    10240,
    11008,
    13824,
    14336,
    15360,
    22016,
    24576,
    27392,
    27648,
    32000,
    32256,
    32512,
    32768,
    33024,
    36864,
    43264,
    49152,
    64000,
    64256,
    102400,
    102656,
    128000,
    128256,
]


_BATCH_SIZE_ALIGNMENT = 8
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
DEFAULT_BATCH_SIZES = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 9)
]


def _get_config_file_name(
    op_type: str,
    batchs: int,
    hidden_size: int,
) -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    return (
        f"op_type={op_type},batchs={batchs},hidden_size={hidden_size} "
        + f"device_name={device_name}.json"
    )


def _get_expand_config(hidden_size) -> List:
    default_sn_lst = [1, 2, 4, 8, 16, 32, 64]
    if hidden_size % 256 == 0:
        default_sn_lst.extend([128, 256])
    elif hidden_size % 128 == 0:
        default_sn_lst.append(128)
    configs = []
    for block_n in [32, 64, 128, 256, 512, 1024]:
        for split_n in default_sn_lst:
            for num_warps in [4, 8]:
                configs.append(
                    {
                        "BLOCK_N": block_n,
                        "SPLIT_N": split_n,
                        "num_warps": num_warps,
                    }
                )
    return configs


def _get_shrink_config(hidden_size) -> List:
    default_sk_lst = [1, 2, 4, 8, 16, 32, 64]
    if hidden_size % 256 == 0:
        default_sk_lst.extend([128, 256])
    elif hidden_size % 128 == 0:
        default_sk_lst.append(128)
    configs = []
    for block_k in [32, 64, 128, 256, 512, 1024]:
        for split_k in default_sk_lst:
            for num_warps in [4, 8]:
                configs.append(
                    {
                        "BLOCK_K": block_k,
                        "SPLIT_K": split_k,
                        "num_warps": num_warps,
                    }
                )
    return configs


def _generate_data(
    batchs,
    hidden_size,
    lora_nums,
    max_rank,
    max_length,
    dtype,
    op_type,
    device,
):
    seq_len_tensor = torch.randint(max_length, max_length + 1, (batchs,)).to(
        device
    )
    total_tokens = seq_len_tensor.sum()
    if op_type == "shrink":
        inputs_tensor = torch.rand(
            (total_tokens, hidden_size),
            dtype=dtype,
            device=seq_len_tensor.device,
        )
        lora_weights = torch.rand(
            (lora_nums, max_rank, hidden_size),  # col-major
            dtype=dtype,
            device=seq_len_tensor.device,
        )
        # NOTE  shrink kernel using torch.float32 as output type
        our_out_tensor = torch.zeros(
            (total_tokens, max_rank),
            dtype=torch.float32,
            device=seq_len_tensor.device,
        )
    else:
        inputs_tensor = torch.rand(
            (total_tokens, max_rank),
            dtype=dtype,
        ).to(device)
        lora_weights = torch.rand(
            (lora_nums, hidden_size, max_rank),  # col-major
            dtype=dtype,
            device=seq_len_tensor.device,
        )
        our_out_tensor = torch.rand(
            (total_tokens, hidden_size),
            dtype=dtype,
            device=seq_len_tensor.device,
        )

    lora_indices_tensor = torch.randint(
        0,
        lora_nums - 1 if lora_nums > 1 else 1,
        (batchs,),
        device=seq_len_tensor.device,
    )
    return (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        lora_indices_tensor,
    )


def run_timing(
    method: Callable,
    num_calls: int,
    batchs: int,
    hidden_size: int,
    config: dict,
    op_type: str,
    device: str = "cuda",
) -> float:
    """
    Based on cuda graph to eliminate the host overhead
    """
    num_loras = 64
    max_rank = 64
    dtype = torch.float16
    max_seq_lengtgh = 1  # decoding stage
    (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        lora_indices_tensor,
    ) = _generate_data(
        batchs,
        hidden_size,
        num_loras,
        max_rank,
        max_seq_lengtgh,
        dtype,
        op_type=op_type,
        device=device,
    )

    def run():
        method(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            lora_indices_tensor,
            override_config=config,
        )

    # JIT compilation & warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    # Capture 100 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for i in range(num_calls):
        torch.cuda.synchronize()
        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_calls * 100) * 1000  # us
    graph.reset()
    return avg


def run_grid_bgmv(
    method: Callable,
    batchs: int,
    hidden_size: int,
    op_type: str,
):
    num_calls = 10
    best_config = None
    best_time_us = 1e20
    configs = (
        _get_expand_config(hidden_size)
        if op_type == "expand"
        else _get_shrink_config(hidden_size)
    )

    for config in tqdm(configs):
        try:
            kernel_dur_us = run_timing(
                method=method,
                num_calls=num_calls,
                batchs=batchs,
                hidden_size=hidden_size,
                op_type=op_type,
                config=config,
            )
            # tqdm.write(
            #         f"{batchs=} {hidden_size=} "
            #         f"{kernel_dur_us=:.1f} "
            #         f"{config=}"
            #     )
            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us
                tqdm.write(
                    f"best config: "
                    f"{batchs=} {hidden_size=} "
                    f"{kernel_dur_us=:.1f} "
                    f"{config=}"
                )
        except triton.runtime.autotuner.OutOfResources:
            continue
    print("best_time_us", best_time_us)
    print("best_config", best_config)

    filename = _get_config_file_name(op_type, batchs, hidden_size)

    print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = json.load(f)
    existing_content[f"batchs={batchs},hidden_size={hidden_size}"] = best_config
    sava_path = os.path.join(SAVE_DIR, filename)
    with open(sava_path, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")


def generate_configs(op_type: str):
    method: Callable = bgmv_expand if op_type == "expand" else bgmv_shrink
    for batchs in DEFAULT_BATCH_SIZES:
        for hidden_size in HIDDEN_SIZE:
            run_grid_bgmv(
                method,
                batchs,
                hidden_size,
                op_type=op_type,
            )
