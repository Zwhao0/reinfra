# GREYHOUND 交接（v1）

## 当前状态

- 两台 8-GPU H800 服务器已完成 GREYHOUND 双机慢节点定位验证。
- 计算慢节点：在主机 2 的 GPU 0（global rank 8）注入减速后，控制器判定 `comp`，并定位到 rank 8。
- 通信慢节点：在 rank 0 注入仅用于测试的 NCCL 延迟后，控制器判定 `comm`；该注入默认关闭，实验结束后两机均已清理。
- `single_comm.py` 的双机 IB 通信压测已通过（200 MiB，约 28.8 GiB/s）。
- README 的核心安装、单机 8-GPU 冒烟、双机计算/通信定位路径已跑通；S2/S3/S4 缓解策略不在本轮验收范围。

## 目录与版本

- 本机代码根目录：`infra/slow node detection/GREYHOUND`
- 两台服务器代码根目录：`/home/test/weihao/slow node detection/GREYHOUND`
- 实验原始日志：本机 `reproduction-data/greyhound-20260923/`
- 操作与复现指南：`REPRODUCTION_ZH.md`
- GitHub：`https://github.com/Zwhao0/infra`；固化版本：tag `v1`（“GREYHOUND 测试成功”）。
- 同步范围：代码、配置和文档；`trainlog/`、注入生成的 `.pkl` 与实验中间结果留在产生它们的机器上。

## 下次开发从这里开始

1. 先阅读 `REPRODUCTION_ZH.md`，其中有启动容器、单机/双机命令、日志位置和清理步骤。
2. 在两台主机启动 `greyhound-ae` 容器后，再按文档执行训练或定位实验。
3. 修改代码后：在本机 `infra` 提交并推送 GitHub；服务器无外网时，将本机提交中的 GREYHOUND 代码用 `git archive` 经 SSH 传到两机，再比对哈希。服务器的 GREYHOUND 子目录仍有独立的上游 Git 历史，不要将它直接与 `infra` 的 Git 历史合并。

## 注意

- 测试用通信延迟开关只有同时设置环境变量和容器内标记文件才生效；日常实验必须保持关闭。
- 服务器路径目前保留 `weihao`，因为容器启动脚本依赖该挂载路径；本机目录和 GitHub 仓库已统一为 `infra`。
