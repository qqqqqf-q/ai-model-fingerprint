# AI 模型随机数指纹数据集

通过"随机选数字"的统计分布指纹，区分 / 检测 17 个 AI 模型。原理源自 [hlwy-ai-checker](https://github.com/hanlinwenyuan/hlwy-ai-checker)。

## 原理
大模型并非真随机数生成器。固定 prompt「请从 1 到 355 之间随机选择一个数字」、`temperature=1.0` 大量采样，不同模型因训练数据 / 架构 / RLHF / tokenizer 差异，会产生统计上可区分的分布指纹。该指纹不易被系统提示词覆盖，可用于检测第三方 API 是否掺假。

## 采集
- OpenRouter API，17 个模型，每模型 ~300 有效样本（minimax 后处理恢复 622）
- prompt 固定，`max_tokens=32`，大多数模型用 `reasoning.enabled=false` 禁用推理
- `grok-build-0.1` / `kimi-k2.7-code` 强制 reasoning，改用 `max_tokens=2048`
- **每条请求完整保存**：raw response body / `system_fingerprint` / `usage`(含 cost) / request 参数 / latency / http_status
- 支持断点续采、并发、429 退避重试

## 数据结构
| 路径 | 内容 |
|---|---|
| `data/raw_*.jsonl` | 每模型全量原始记录（一行一条请求，含完整 response） |
| `data/summary.json` | 指纹向量 + 统计量 + 17×17 相似度矩阵 + 全量数字序列 |
| `figures/` | 可视化图表（分布 / 统计量 / 相似度热力图 / 箱线图 / 综述 overview） |

## 模型（17）
claude opus 4.6/4.7/4.8、sonnet 4.6、haiku 4.5；gpt-5.5/5.4/4o-mini；glm-5.2；deepseek v3.2/v4-flash/v4-pro；minimax-m3；qwen3.7-plus；grok-build-0.1；kimi k2.6/k2.7-code

## 关键发现
- **claude opus 4.6 ≈ 4.7**（cos=0.997，指纹几乎相同），但 **4.8 完全不同**（cos≈0.04），且 4.8 `unique=2` 准固定输出（300 次几乎只输出 237）
- **kimi k2.6 vs k2.7-code**（同 base、k2.7 为后训练版）：cos=0.631，后训练明显改变指纹
- **deepseek 同栈不同语料** v3.2 / v4-flash / v4-pro 互相 cos=0.15~0.31
- 跨模型最像：haiku-4.5 ≈ sonnet-4.6 (0.997)、kimi-k2.7-code ≈ glm-5.2 (0.916)

## 复现
```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml   # 填入 OpenRouter key
python run.py --samples 300          # 采集
python check_data.py                 # 完整性验证
python analyze.py                    # 统计 + 图表
python estimate.py                   # token / cost 预估
```

## 成本
全量采集约 **$1.03**（grok reasoning 占大头，637 output token/req）。
