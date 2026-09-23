# GREYHOUND 两机复现与使用指南

更新时间：2026-09-23。本指南只以项目 `README.md` 和本机实测结果为验收标准。

## 1. 当前结论

README 的安装、编译、单机 8-GPU、双机 16-GPU、双机 pre-check、计算慢节点检测、根因验证和 S2 micro-batch 调整已经跑通。已在主机 2 / GPU0 注入计算退化，GREYHOUND 最终正确定位到 global rank 8，并通过独立计算验证判定根因为 `comp`。通信压测通过 TCP 路径跑通，但独立的 200 MB InfiniBand 点对点测试仍报 `vendor err 249`；S3/S4 尚未复现。

| README 项目 | 状态 | 说明 |
|---|---:|---|
| Docker 安装与 8 张 H800 可见 | 通过 | 两台服务器的 `greyhound-ae` 容器均正常 |
| detector `.so` 与 controller wheel | 通过 | 两台机器均已编译 |
| `run_training.py` 单机测试 | 通过 | 两机分别 3 次迭代 |
| `run_training_dp.py` 单机测试 | 通过 | 50/300 次迭代测试均完成训练 |
| 双机联合训练 | 通过 | 16 GPU；正常训练及长时间定位实验均已运行 |
| GPU 降频注入 | 通过 | GPU0 锁到 900 MHz，均值约增加 4.4% |
| `single_comm.py` 通信压测 | 部分通过 | IB 失败；强制 Socket 后成功 |
| 双机 pre-check | 通过 | 16 个 rank 完成计算和跨节点通信验证，`precheck_done=1` |
| 双机计算慢节点定位 | 通过 | 主机 2 / GPU0 / global rank 8 定位正确，根因为 `comp` |
| S2 micro-batch 缓解 | 通过但有长跑限制 | rank 8 从 2 降到 1；缓解有效，但单次 dataloader 会提前耗尽 |
| S3 通信重配置 | 未复现 | 需要稳定的通信慢节点注入 |
| S4 checkpoint-restart | 未复现 | 需要准备可恢复 checkpoint 并触发策略 |

## 2. 固定环境

- 主机 1：`host1`，`10.10.4.1`
- 主机 2：`host2`，`10.10.4.2`
- 服务器代码：`/home/test/weihao/slow node detection/GREYHOUND`
- 容器：`greyhound-ae`
- 容器代码：`/workspace/Greyhound`
- 本机代码：`/Users/henry/Desktop/故障定位和恢复/reinfra-sync/slow node detection/GREYHOUND`
- 本机原始数据：`/Users/henry/Desktop/故障定位和恢复/reproduction-data/greyhound-20260923`
- 汇总数据：`/Users/henry/Desktop/故障定位和恢复/reproduction-data/greyhound-20260923/metrics.csv`

SSH 已配置免密别名。主机 2 执行 `sudo` 时仍可能要求输入服务器密码；密码只查本机服务器文档，不要写进代码或 Git。

## 3. 日常进入方法

```bash
ssh host1
sudo docker exec -it greyhound-ae bash
cd /workspace/Greyhound
```

主机 2 把 `host1` 换成 `host2`。退出容器输入 `exit`，再输入一次退出 SSH。

先做健康检查：

```bash
sudo docker ps --filter name=greyhound-ae
sudo docker exec greyhound-ae nvidia-smi -L
```

应看到容器为 `Up`，并看到 8 张 GPU。

## 4. 首次编译或代码更新后重编译

在两台服务器的容器内分别执行：

```bash
cd /workspace/Greyhound/detector
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"

cd /workspace/Greyhound/detector
python setup.py bdist_wheel
python setup.py install
```

检查产物：

```bash
ls -l /workspace/Greyhound/detector/build/libncclprobe.so
ls -l /workspace/Greyhound/detector/dist/*.whl
```

## 5. 单机 8-GPU 冒烟测试

进入任一容器后执行：

```bash
cd /workspace/Greyhound
python run_training.py \
  --iter 3 \
  --master 127.0.0.1 \
  --master-port 29600 \
  --logdir /workspace/Greyhound/trainlog/manual-single
```

看到 `iteration 3/3`、验证集结果，且无 Python traceback，即训练通过。短作业结束后控制器可能继续等待预检；确认训练结果已输出后可按 `Ctrl-C`。再次运行时换一个未占用的 `--master-port`。

## 6. 双机 16-GPU 快速测试

打开两个本机终端。先启动主机 2，再在 30 秒内启动主机 1。

终端 A：

```bash
ssh host2
sudo docker exec -it greyhound-ae bash
cd /workspace/Greyhound
python run_training_dp.py \
  --iter 100 --nnodes 2 --rank 1 \
  --master 10.10.4.1 --master-port 29610 \
  --logdir /workspace/Greyhound/trainlog/manual-two-node
```

终端 B：

```bash
ssh host1
sudo docker exec -it greyhound-ae bash
cd /workspace/Greyhound
python run_training_dp.py \
  --iter 100 --nnodes 2 --rank 0 \
  --master 10.10.4.1 --master-port 29610 \
  --logdir /workspace/Greyhound/trainlog/manual-two-node
```

两边的 `nnodes`、`master`、`master-port` 必须相同，`rank` 必须分别为 0 和 1。本次实测第 20–100 次迭代均值为 `186.394 ms`，约 `321.90 iter/min`。

### 双机 rank 映射

每台机器由 `torchrun --nproc-per-node 8` 启动 8 个进程，因此：

```text
global_rank = node_rank * 8 + local_gpu_id
```

| 机器 | node rank | global rank | GPU |
|---|---:|---:|---:|
| 主机 1 / `10.10.4.1` | 0 | 0–7 | 0–7 |
| 主机 2 / `10.10.4.2` | 1 | 8–15 | 0–7 |

因此主机 2 / GPU0 对应 global rank 8。

### 双机计算慢节点定位实测

双机训练和 `precheck_done=1` 后，在主机 2 宿主机执行：

```bash
sudo nvidia-smi -i 0 -lgc 345
```

由于 H800 的最低硬件频率只造成约 5%–6% 退化，低于代码的 10% 检测阈值，本次还使用仓库自带的模型 hook。在主机 1 容器中执行：

```bash
redis-cli -h 10.10.4.1 set delay_time_8 0.1
```

实测结果：

- local controller 报告 rank 8 发生 fail-slow；其他 rank 也会因同步等待产生连带上报。
- validation 中 rank 8 的计算时间为 `144.192 ms`，其余 rank 为约 `25.6–25.8 ms`。
- global controller 输出 `Reason of fail-slow is comp`。
- S2 计划为 `[2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,3]`，即减少 rank 8、增加 rank 15。
- 正常阶段中位数 `174.7 ms`；注入后、缓解前 `445.0 ms`；S2 后 `257.9 ms`；完全恢复后 `174.8 ms`。

恢复命令必须全部执行：

```bash
# 主机 1 容器内
redis-cli -h 10.10.4.1 set delay_time_8 0
redis-cli -h 10.10.4.1 set batch_distribution \
  '[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]'
redis-cli -h 10.10.4.1 set dp_version 2

# 主机 2 宿主机
sudo nvidia-smi -i 0 -rgc
```

本次原始日志位于本机：

```text
/Users/henry/Desktop/故障定位和恢复/reproduction-data/greyhound-20260923/
  node1/readme-two-node-localize/
  node2/readme-two-node-localize/
```

## 7. 按 README 注入计算慢节点

先让训练稳定运行，再在宿主机的另一个 SSH 终端执行：

```bash
sudo nvidia-smi -i 0 -lgc 900
sleep 15
sudo nvidia-smi -i 0 -rgc
```

无论实验是否异常，都必须执行最后一条恢复命令。可用下列命令确认没有残留锁频：

```bash
sudo nvidia-smi -i 0 -rgc
nvidia-smi -i 0 --query-gpu=clocks.current.sm --format=csv,noheader
```

本次结果：注入前 73 次迭代均值 `312.334 ms`；900 MHz 注入期间 47 次迭代均值 `326.011 ms`，增加约 `4.38%`。H800 与 README 示例中的 A10 不同，频率与减速比例不能直接照搬。

## 8. 按 README 运行通信压测

`single_comm.py` 需要两个 rank。当前 IB 路径会报 `vendor err 249`，所以可先用 Socket 验证脚本。仍然打开两个终端，先 rank 1，再 rank 0：

```bash
# 主机 2 / 容器内
cd /workspace/Greyhound/detector/injection
MASTER_ADDR=10.10.4.1 MASTER_PORT=29702 WORLD_SIZE=2 RANK=1 \
NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
python single_comm.py --tensor-size 20 --duration 1 \
  --logdir /workspace/Greyhound/trainlog/manual-comm
```

```bash
# 主机 1 / 容器内
cd /workspace/Greyhound/detector/injection
MASTER_ADDR=10.10.4.1 MASTER_PORT=29702 WORLD_SIZE=2 RANK=0 \
NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
python single_comm.py --tensor-size 20 --duration 1 \
  --logdir /workspace/Greyhound/trainlog/manual-comm
```

本次 Socket 测试成功，发送端记录 `1827.04 MB/s`，接收端记录 `5687.05 MB/s`。两台宿主机时钟未严格同步，且脚本的发送/接收计时方式不同，因此只把它作为“路径跑通”证据，不把两个数当作精确链路带宽。

## 9. 查看结果

```bash
cd /workspace/Greyhound
find trainlog -maxdepth 3 -type f | sort
grep -R "iteration .*iteration (ms)" trainlog/<本次目录> | tail -20
grep -R -E "Pre-Check|Fail-slow|profil|validation|Mitigating|Root cause" \
  trainlog/<本次目录>
grep -R -E "Traceback|NCCL error|vendor err|Unhandled exception" \
  trainlog/<本次目录>
```

重点文件：

- `runner*.log`：训练是否完成、每次迭代耗时、异常。
- `global_controller_*.log`：预检、全局定位、验证和缓解。
- `local_controller_*.log`：每个节点的模式识别和上报。
- `ncclprobe.log`：NCCL 拦截记录。
- `megatron_output_*.log`：Megatron 训练输出。

若要测 detector 开销，使用当前新增的开关运行相同参数：

```bash
python run_training_dp.py --iter 50 --disable-detector \
  --master-port 29620 --logdir /workspace/Greyhound/trainlog/overhead-off
python run_training_dp.py --iter 50 \
  --master-port 29621 --logdir /workspace/Greyhound/trainlog/overhead-on
```

本次短测 detector off/on 均值分别为 `324.039/316.248 ms`，表观开销 `-2.40%`。这是短测噪声，不代表 detector 会加速；只能说明没有观察到明显正开销。

## 10. 当前已知限制

- 变更后的 503-byte NCCL 控制标记是 pre-check 能运行的必要修复；两台服务器必须使用同一版本的 `microbatches.py`。
- H800 的最低 345 MHz 降频不足以稳定超过 10% 阈值，定位测试需配合仓库自带的 `delay_time_<global_rank>` hook。
- S2 不均匀 micro-batch 会让各 rank 以不同速度消耗当前 `single` dataloader。此次 1200 次迭代实验在第 1085 次提前抛出 `StopIteration`。定位、validation、S2 和恢复均在此之前完成，但长时间使用 S2 前必须修复动态数据分片或改用经过验证的 cyclic dataloader。
- S2 期间 Megatron 的 `global batch size` 日志按当前输出 rank 的局部 micro-batch 推算，会显示 192 等值；不能直接当作所有 rank 的真实总和。真实总量应按分配数组之和乘以 micro-batch size 计算。
- 独立 `single_comm.py --tensor-size 200` 的 IB 点对点路径仍有 `vendor err 249`；训练和 pre-check 的 IB 路径本次可以正常完成。

## 11. 下一步：用于真实集群慢节点定位

先不要自动调整生产训练。建议分三阶段：

1. **影子观测**：将 `libncclprobe.so` 注入真实训练，只采集 NCCL 事件、迭代耗时以及 rank→节点→GPU 映射；告警但不缓解。
2. **校准定位**：在两台机器的不同 GPU 上重复注入，统计正确定位、误报、漏报和检测延迟；并修复动态数据分片和独立 IB 点对点错误。
3. **受控缓解**：先接入人工确认、checkpoint 和调度器；稳定后才分批开启 micro-batch 调整或重排。所有动作都要有超时、回滚、审计和“一键关闭”。

当前已经证明主机 2 / GPU0 / rank 8 的计算慢节点可以被定位。下一步是分别在两台机器的不同 GPU 上重复该实验，并完成通信型慢节点定位、S3 和 S4。
