import torch
import time
import subprocess
import re
import argparse


def get_free_vram_gb(gpu_id: int = 0) -> float:
    """通过 nvidia-smi 查询指定 GPU 的空闲显存（GB）"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", f"--id={gpu_id}"],
        capture_output=True, text=True
    )
    free_mb = float(result.stdout.strip())
    return free_mb / 1024.0


def reserve_vram(size_in_gb: float = 16.0):
    """预分配指定大小的显存到 PyTorch 缓存池，防止被其他进程抢占。"""
    if not torch.cuda.is_available():
        return False
    num_elements = int(size_in_gb * (1024 ** 3) / 4)
    try:
        dummy = torch.empty(num_elements, dtype=torch.float32, device="cuda")
        del dummy
        print(f"[VRAM] Reserved {size_in_gb:.2f} GB in PyTorch caching allocator.")
        return True
    except RuntimeError:
        print(f"[VRAM] Failed to reserve {size_in_gb:.2f} GB — not enough free VRAM.")
        return False


def occupy_gpu(gpu_id: int = 0, check_interval: float = 5.0, margin_mb: float = 256.0):
    """
    持续监控并占据指定 GPU 的所有可用显存。

    Args:
        gpu_id: GPU 编号
        check_interval: 检查间隔（秒）
        margin_mb: 保留的余量（MB），避免完全占满导致系统不稳定
    """
    torch.cuda.set_device(gpu_id)

    # 先做一次初始占据
    free_gb = get_free_vram_gb(gpu_id)
    margin_gb = margin_mb / 1024.0
    initial_reserve = free_gb - margin_gb
    if initial_reserve > 0.1:
        print(f"[Init] GPU {gpu_id} 空闲 {free_gb:.2f} GB，尝试占据 {initial_reserve:.2f} GB ...")
        reserve_vram(initial_reserve)

    print(f"\n[Monitor] 开始监控 GPU {gpu_id}，每 {check_interval}s 检查一次，保留余量 {margin_mb:.0f} MB")
    print("[Monitor] 按 Ctrl+C 退出\n")

    try:
        while True:
            time.sleep(check_interval)
            free_gb = get_free_vram_gb(gpu_id)

            if free_gb > margin_gb + 0.1:
                grab = free_gb - margin_gb
                # print(f"[Grab] 检测到 {free_gb:.2f} GB 空闲，尝试占据 {grab:.2f} GB ...")
                reserve_vram(grab)
            else:
                print(f"[OK] GPU {gpu_id} 空闲 {free_gb:.2f} GB，已充分占据。")

    except KeyboardInterrupt:
        print("\n[Exit] 停止监控。显存将在进程退出后释放。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU 站卡程序 - 持续占据指定 GPU 的空闲显存")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号 (默认: 0)")
    parser.add_argument("--interval", type=float, default=10.0, help="检查间隔秒数 (默认: 5)")
    parser.add_argument("--margin", type=float, default=2560.0, help="保留余量 MB (默认: 256)")
    args = parser.parse_args()

    print(f"=== GPU 站卡程序 ===")
    print(f"目标 GPU: {args.gpu}")
    print(f"检查间隔: {args.interval}s")
    print(f"保留余量: {args.margin} MB\n")

    occupy_gpu(gpu_id=args.gpu, check_interval=args.interval, margin_mb=args.margin)