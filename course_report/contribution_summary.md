# Dataset + Task + Model：项目成果与评分维度

作者：Xiangwen Wang、Qichang Liu · 更新：2026-11-26

**最终选择：Option 4，Dataset + Task + Model（Full Pipeline）。** LifeSnaps是PyHealth贡献的数据集，PMData用于独立研究复现。核心贡献包含数据加载、任务定义、模型、端到端训练评估、文档测试与实际消融。以下配分是rubric中的项目权重，不是已经获得的分数。

本文件按评价维度整理已完成工作及其证据；[完整研究报告](README.md)将同一组成果组织为连贯的研究与展示叙述。

## 1. Core Implementation：20分

这20分是核心实现的评分维度。Dataset、Task和Model的对应分表不简单累加成60分。

| 对应项目 | 已实施工作 | 验证结果或结论 |
|---|---|---|
| Dataset实现 | LifeSnapsDataset继承BaseDataset；原始CSV准备、完整日/去重/缺失处理；YAML定义小时及每日事件 | 实际数据得到71人、5,850完整日、228标签日；合成测试核验读取、事件和异常路径 |
| Task实现 | StressDetectionLifeSnaps继承BaseTask；participant-day样本；5×24 sensor、22维context和二分类标签；无标签预训练模式 | 监督3样本、预训练4完整日的fixture结果符合预期；STAI未进入输入 |
| Model实现 | WearableStressModel继承BaseModel；双分支编码、融合、对比目标和分类；probe/fine-tune/checkpoint | 形状、梯度、冻结行为、目标公式和保存恢复通过测试；完整真实训练已执行 |
| 文档与类型 | Google-style参数/返回/异常说明、类级使用示例、方法类型标注 | Dataset三个方法的参数和返回标注已核对；组件API文档提供输入输出说明 |
| 代码质量 | 模块职责分离；snake_case/PascalCase；针对性Ruff lint/format | 已有贡献代码检查记录；Dataset行长和命名复查通过，最长87字符 |

错误处理覆盖未知非空STAI类别和无完整日等情况。缺文件、坏列仍主要使用pandas底层异常；全量静态类型检查、所有输入边界与所有运行环境不在已有验收结论之内。

## 2. Required File Updates：12分

| 子项与配分 | 实际交付 |
|---|---|
| 核心文件，3分 | Dataset、Task、Model及lifesnaps.yaml |
| API RST，3分 | 三个组件各自的API文档 |
| 索引，2分 | docs/api目录索引；另包含Python模块公开导出 |
| Example，1分 | 完整CLSP训练example、MLP配置example及分析入口 |
| Tests，3分 | Dataset/Task、Model、训练辅助和分析测试 |

记录基点到当前代码提交29c5287包含35个贡献变更文件。论文复现说明、实验参数、测量值和findings与代码一并保存在fork。课程综合报告位于独立分支，不属于代码贡献分支。

## 3. Test Cases：11分

| 子项 | 设计与实际验证 |
|---|---|
| 小型合成数据，2分 | 四名虚拟参与者、临时CSV、最小模型张量；不用真实数据或下载BERT |
| 运行速度，2分 | 最近套件47项＋3个subtests通过，14.82秒含启动；最慢call0.26秒、setup0.68秒 |
| 临时目录和清理，1分 | fixture使用临时目录，包含setup失败时的清理验证 |
| 覆盖范围，6分 | 加载/事件/Task特征标签、缺失/重复/无标签、forward/gradient、probe冻结、checkpoint、split/normalization、bootstrap/AUC分解及数据/cache回归 |

结果中有一条依赖更新提示。测试支持软件按预期处理输入和计算，不是模型预测有效性的证明。Sphinx完整构建、多worker执行尚未验证。

## 4. Ablation Study / Example Usage：5分

Dataset/Task路线的要求是特征或任务配置变化、用现有PyHealth模型展示性能比较，并记录设置与findings；我们使用PyHealth MLP直接落实这组实验。完整CLSP路线进一步加入目标和文字机制控制及日历干预。CLI有合成短流程，真实结果由真实数据运行产生。

| 实验 | 已完成的受控比较 | 实测结果与结论 |
|---|---|---|
| MLP输入模态 | both/sensors/context，三seed；固定同seed划分，各配置重训 | AUC69.50±3.57 / 68.19±1.70 / 55.51±18.08%；sensor-only与双模态均值接近，context-only波动大 |
| MLP观测偏移 | offset1/7，按历史观测数量移动context并重训 | 69.33±0.89 / 70.79±2.29%；未呈单调下降，属于训练任务变化 |
| 日历context干预 | k0/1/3/7；仅改变动态context，固定模型和测试行 | supervised同日52.83±3.51%、7日50.00±6.04%；reference同日50.03±7.65%、7日51.44±10.18%；全部下降差配对CI包含0，未建立CLSP更抗过期的证据 |
| 文本机制 | pretrained frozen BERT / random frozen BERT / statistics；同seed、数据和预算，分别重训 | 50.03±7.65 / 52.50±19.80 / 45.13±8.82%；未建立稳定预训练语言收益，统计控制还改变了表示和缩放 |
| 对比目标 | reference / stop_gradient / one_hot；同seed保持其他条件 | 50.03±7.65 / 50.08±12.98 / 47.55±9.17%；未发现稳定优胜者，reference用于对齐目标梯度行为 |
| AUC诊断 | 固定预测，按同人/跨人正负配对分解 | 93–94%的配对来自不同人；总体AUC不足以证明个体每日跟踪能力 |

±均为三seed总体SD，不是CI。MLP和CLSP协议不同，其绝对分数不是直接模型排名。AUC分解是诊断、bootstrap是统计方法，不单独算作新的训练消融。CLSP有9次目标训练和6次新增文字训练，共15次；MLP另有15次。真实日历实验没有触发fallback，合成测试覆盖该边界。

模型细项列举了超参数变化和现有数据集实验；我们实际报告的是目标/文字机制控制以及新接入LifeSnaps上的完整流程，没有将其冒充lr/hidden-dimension网格搜索。Full Pipeline的具体评分归属和消融novelty由教师判定。普通模态变化本身不作为新增研究创新的唯一依据。

## 5. Pull Request Formatting：7分

PR描述材料已包含作者姓名/NetID、Dataset+Task+Model贡献类型、论文链接、实现概述、文件说明和验证结果。两位作者为Xiangwen Wang（xw120）和Qichang Liu（qichang4）。

代码已经推送到qichang-liu/PyHealth的lifesnaps-stress-detection分支；正式上游PR尚未创建，因此描述材料准备完成不等于这一提交环节已经完成。外部CI与维护者审查也尚未发生。

## 6. Full Pipeline：额外10分资格

Dataset、Task、Model、文档和测试均包含在代码贡献中。完整example已经运行了数据加载→任务处理→模型训练→评估，并支持合成短流程；真实BERT配对、预训练、probe、fine-tune和checkpoint恢复已经接通。模型实现承接原论文提出的方法，并非仅用已有MLP训练新数据。

这些成果支持按Option4申请额外学分，但不代表已获得10分。模型novelty、贡献完整性与最终PR状态仍由课程评定；原论文算法归作者，本组贡献是其PyHealth实现、工程验证和进一步实验分析。

## 7. Video Presentation：10分

| 评分项 | 本组已有内容和结论 |
|---|---|
| 4–7分钟，2分 | 文字与slides素材已有；实际录制时长和公开视频链接尚未完成 |
| General problem，2分 | 标签稀缺条件下，使用wearable序列和context进行每日压力分类 |
| Specific approach，2分 | Transformer＋context残差编码、同源文字/BERT、对比预训练、probe和fine-tune |
| Reproduction，2分 | 两数据集40次修正CLSP运行；LifeSnaps73.84 vs监督74.37、PMData71.74 vs73.98概率AUC%；未建立原报告的相对收益，并披露协议差异 |
| Extensions/ablations，2分 | 上述实验的动机、控制变量、数值、不确定性和结论；区分模型消融与评价诊断 |

正文采用研究叙述呈现这些内容，实际录制和提交仍是待完成环节。

## 8. Final GenAI Form：15分

数据处理、算法实现、训练循环、结果生成与分析四方面的完整回答及真实prompt/response总结已在本地整理。个人评分、时间、账号字段和提交回执尚不能代替本人填写或认定已提交；个人AI日志未推送到公开报告分支。

## 9. 统一结论

工程结果是一个实际运行、可测试、带文档与实验记录的Dataset + Task + Model全流程。科学结果是在当前固定协议与有限预算下，CLSP未显示稳定预训练优势，文字机制与旧context鲁棒性也未得到稳定支持；同人/跨人分析进一步限定了总体AUC的解释范围。两类结果共同构成项目成果。

课程将PR名义总分标为50，但上述PR五个细项合计55；本文件按原条目展示权重，不自行换算或估算已得分。正式PR、公开视频与个人表单回执尚待完成。

## 实验与实现来源

- [完整研究报告](README.md)
- [贡献README](../examples/wearable/README.md)
- [完整训练example](../examples/wearable/lifesnaps_stressdetection_wearable_stress.py)
- [MLP实验设置与findings](../examples/wearable/lifesnaps_stressdetection_findings.md)
- [目标函数对照](../examples/wearable/lifesnaps_objective_ablation.md)
- [日历、文本与AUC分析findings](../examples/wearable/lifesnaps_proposal_findings.md)

以上为依据课程rubric和已保存的实现/实验材料整理的成果说明，配分不等于教师评分。
