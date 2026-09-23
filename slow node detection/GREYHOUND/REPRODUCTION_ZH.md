# GREYHOUND 两机复现与使用指南

更新时间：2026-09-23。本指南以项目 `README.md` 为验收标准；论文数字只作辅助参考。

## 1. 当前结论

README 的安装、编译、单机 8-GPU 训练、双机 16-GPU 训练、计算降速注入、日志生成和开销 A/B 测试已经跑通。通信压测通过 TCP 路径跑通，但原生 InfiniBand 点对点测试报 `vendor err 249`。短作业能启动 GREYHOUND 预检并识别 DP clique，但预检没有在作业结束前完成，因此不能声称已经复现自动定位、根因验证及 S2/S3/S4 缓解闭环。

| README 项目 | 状态 | 说明 |
|---|---:|---|
| Docker 安装与 8 张 H800 可见 | 通过 | 两台服务器的 `greyhound-ae` 容器均正常 |
| detector `.so` 与 controller wheel | 通过 | 两台机器均已编译 |
| `run_training.py` 单机测试 | 通过 | 两机分别 3 次迭代 |
| `run_training_dp.py` 单机测试 | 通过 | 50/300 次迭代测试均完成训练 |
| 双机联合训练 | 通过 | 16 GPU、100 次迭代完成 |
| GPU 降频注入 | 通过 | GPU0 锁到 900 MHz，均值约增加 4.4% |
| `single_comm.py` 通信压测 | 部分通过 | IB 失败；强制 Socket 后成功 |
| 预检、自动定位和根因验证 | 未完整通过 | 短作业结束时仍在等待 pre-check |
| S2/S3/S4 自动缓解 | 未复现 | 需要先解决预检和 IB，再运行更长作业 |
| 论文规模准确率/吞吐提升 | 未复现 | 我们只有 16 GPU，论文主实验规模和工作负载不同 |

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

本次短测 detector off/on 均值分别为 `324.039/316.248 ms`，表观开销 `-2.40%`。这是短测噪声，不代表 detector 会加速；只能说明没有观察到明显正开销。论文报告平均跟踪开销约 `0.39%`、最高 `1.1%`，我们的样本不足以验证该精确数字。

## 10. 论文数据只作辅助参照

论文报告生产数据 499 个作业中 498 个定位正确、跟踪平均开销约 0.39%，以及缓解后相对“不处理”约 1.58 倍吞吐。论文使用最多 256 张 H800 和不同工作负载；我们的 16-GPU 短测没有测定位准确率或 S2/S3/S4，因此不应宣称结果与论文等价。参考：[USENIX 论文页面](https://www.usenix.org/conference/atc25/presentation/wu-tianyuan)；[论文 PDF](https://www.usenix.org/system/files/atc25-wu-tianyuan.pdf)。

## 11. 下一步：用于真实集群慢节点定位

先不要自动调整生产训练。建议分三阶段：

1. **影子观测**：将 `libncclprobe.so` 注入真实训练，只采集 NCCL 事件、迭代耗时以及 rank→节点→GPU 映射；告警但不缓解。
2. **校准定位**：先修复 IB `vendor err 249` 和单机预检耗时问题，在长作业中人为降频、限带宽，验证能稳定指出正确节点/GPU，并统计误报、漏报、检测延迟和开销。
3. **受控缓解**：先接入人工确认、checkpoint 和调度器；稳定后才分批开启 micro-batch 调整或重排。所有动作都要有超时、回滚、审计和“一键关闭”。

离真实上线最近的下一个实验是：双机运行 30–60 分钟的固定训练，在完成 pre-check 后只降频一张 GPU，确认 `global_controller_*.log` 能报告正确 rank，并在恢复频率后回到正常状态。

