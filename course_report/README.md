# 多模态压力检测的复现与扩展研究

作者：Xiangwen Wang、Qichang Liu · CS598 Deep Learning for Healthcare · 整理日期：2026-11-26

**本项目选择课程Option 4：Dataset + Task + Model（Full Pipeline）。** 我们以LifeSnaps为主数据集，将原始记录加载、每日压力任务、多模态模型和训练评估连接成完整PyHealth流程，并以独立复现和消融实验检验文字引导预训练的收益。研究同时考察context时效性、文字表示机制和总体AUC的含义。

我们完成了LifeSnaps的Dataset、Task、Model和完整训练流程，并执行了日历context干预、文本机制对照和同人/跨人AUC分析。在当前实验协议下，结果未显示稳定的CLSP预训练优势，也未建立其对旧context的鲁棒性优势。这些结果将研究重点从总体分数的接近程度，推进到模型机制、使用条件与评价含义的检验。

## 1. 研究背景与问题定义

可穿戴设备每天产生步数、心率等记录，而连续收集压力问卷的成本较高，导致压力标签相对稀少。这种数据不平衡使自监督学习成为一个有吸引力的方向：模型先从大量无标签记录中学习表示，再利用有限的有标签日期完成压力分类。

我们复现的论文是Yang、Yu、Sano的 **Contrastive Pretraining for Stress Detection with Multimodal Wearable Sensor Data and Surveys**，发表于CHIL 2025。研究同时使用随时间变化的传感器数据和表格型背景信息。后者包括人口学、人格、活动和场所等字段，并不全部是压力问卷答案。[论文及作者信息](https://proceedings.mlr.press/v287/yang25a.html)

本项目的预测单位是“一个人一天”。LifeSnaps输入为五个传感器通道的24小时序列，以及22维context；目标是当天STAI类别转换的二分类：Above average为1，Average或Below average为0。我们的输入排除了压力标签及其派生字段，以避免标签泄漏。任务对应当天压力分类；本研究的评估范围不包括未来压力预测或临床诊断有效性。

LifeSnaps是PyHealth贡献的主数据集：71名参与者、5,850个完整日期、228个有标签日期；有标签日期来自其中50人。PMData用于另一套独立复现，传感器和context组成不同；当前没有贡献PMData的PyHealth Dataset/Task，也没有进行跨数据集训练后迁移测试。

## 2. 原论文的多模态对比学习方法

![论文原图 Figure 1：多模态数据与文字视图的对齐框架](handin_report_assets/paper_figure1_framework.png)

图1来源：Yang、Yu、Sano（2025），原论文Figure 1，PDF第3页。图中展示作者提出的模型框架。

第一条分支处理数值输入：传感器Transformer提取小时序列表示，残差网络编码tabular context，两者拼接、融合并投影。第二条分支把同一天的传感器统计量和context写成结构化文字，用冻结BERT得到文字表示，再进行投影。对比学习让两条分支形成相互对应的表示。

例如，一个日期的心率序列和活动信息既以数值形式出现，也可被描述为“heart rate mean …; activity …”。文字与数值分支因此共享底层测量信息。我们根据允许字段构建文字，并排除压力标签。模型采用CLIP风格的对齐目标，文字编码器则使用BERT。

预训练后，先冻结表示模型、只训练分类头（linear probing），再微调模型进行压力分类。BERT只服务于预训练的文字视图；下游分类推理使用sensor/context分支即可。

论文的核心假设是文字引导的无标签多模态预训练能够改善下游压力分类。作者在Table 1中报告了以下AUC结果，单位为百分比：

| 数据集 | 作者Supervised | 作者CLSP |
|---|---|---|
| LifeSnaps | 67.18 ± 7.65 | 72.45 ± 4.70 |
| PMData | 80.81 ± 1.62 | 81.14 ± 1.79 |

作者CLSP相对该监督基线的均值差分别为5.27和0.33个百分点，但其优势并未覆盖所有比较方法。我们的结果分析区分了概率AUC与硬预测AUC；PMData在论文中的具体指标口径仍有待进一步核对，因此与作者数值的比较存在这一限制。

## 3. Dataset + Task + Model全流程贡献

我们的核心实现覆盖三个相互衔接的组件。Dataset提供统一的记录表示，Task定义模型学习的样本和目标，Model完成表示学习和预测；训练example连接这三个组件，执行数据划分、预处理、优化与评估。这一实现对应课程的Core Implementation与Full Pipeline贡献。

### 3.1 Dataset：从LifeSnaps原始记录到PyHealth事件

[LifeSnapsDataset](../pyhealth/datasets/lifesnaps.py)继承BaseDataset。它读取小时级传感器和问卷CSV，按参与者和日期组织数据，去除重复小时，并保留原顺序为0至23的完整日。已有小时行中的缺失传感器值经过插值和补齐；缺少小时记录的日期被排除。缺失压力标签保持无标签，未知非空类别和无完整日输入则触发明确异常。

[配置YAML](../pyhealth/datasets/configs/lifesnaps.yaml)定义小时和每日两张表的文件、参与者ID、时间和属性映射。整理后的记录经由父类成为PyHealth事件，因此后续任务可以复用统一的数据接口。LifeSnaps预处理得到71名参与者、5,850个完整日期和228个有标签日期。

### 3.2 Task：从事件到每日压力预测样本

[StressDetectionLifeSnaps](../pyhealth/tasks/stress_detection_lifesnaps.py)继承BaseTask，将“一个人一天”定义为样本单位。输入是5×24的传感器张量与22维context，监督目标为STAI类别映射的二分类标签。我们排除了STAI及其派生字段，避免目标信息进入数值和文字输入。

Task的监督模式产生完整、有标签的日期；预训练模式包含完整无标签日，不提供压力目标。模态配置支持双模态、sensor-only和context-only，历史观测配置支持Task级消融。这些配置使同一Dataset能够支持不同训练任务，而无需重复实现原始数据加载。

### 3.3 Model：多模态编码、对比学习与压力分类

[WearableStressModel](../pyhealth/models/wearable_stress.py)继承BaseModel，包含sensor Transformer、context残差编码器、融合与投影层、压力分类头。监督阶段返回预测概率，并在提供标签时计算分类loss；预训练阶段对齐多模态表示与外部文字embedding，计算对比loss。模型还实现了冻结probe、fine-tune和checkpoint保存恢复。

我们的模型贡献对应论文的多模态文字对齐方法。Transformer、残差网络和BERT是组成模块；贡献在于将其组合、训练目标与阶段行为接入PyHealth。原算法归于论文作者，作者代码曾作为行为参考。复现中发现的soft-target梯度差异已被修正：默认reference保留目标梯度，stop_gradient与one_hot作为明确对照；loss、梯度和旧checkpoint行为都有对应测试。该修正限于目标梯度，其他预处理和协议差异仍保留在结果解释中。

### 3.4 完整训练与评估

[完整example](../examples/wearable/lifesnaps_stressdetection_wearable_stress.py)实现了原始数据→事件→Task样本→参与者划分→训练集预处理→真实文字/BERT配对→对比预训练→冻结probe→fine-tune→留出评估。分类checkpoint由验证集指标选择，保存恢复后的预测经过一致性检查。监督基线使用相同的多模态结构，从头学习压力分类。

这条流程已经在真实LifeSnaps数据上运行，相关目标和文字控制共完成15次CLSP训练。合成示例则提供无真实数据或BERT下载的短流程验证，其随机文字向量只用于检查软件链路。另一个[MLP example](../examples/wearable/lifesnaps_stressdetection_mlp.py)使用现有PyHealth MLP完成15次特征与Task配置实验，直接检验新Dataset/Task在已有模型中的可用性。

完整模型流程提供了工程上的复用能力；真实实验是否支持预训练收益，则由后续数值比较回答。

## 4. 实验协议与主要复现结果

我们有两条实验线。独立repro追踪两数据集的论文复现；PyHealth example验证库内完整流程、受试者留出和新增消融。两条线回答的问题不同。

| 项目 | 独立研究复现 | PyHealth CLSP pilot |
|---|---|---|
| 数据集 | LifeSnaps、PMData各自训练测试 | LifeSnaps |
| 划分 | participant-day五折；同一人可跨训练/测试；少于5个标签日的人固定进训练 | 参与者级60/20/20；测试参与者不进入训练 |
| 新reference训练数 | 每数据集5折×4seed，共40次 | 目标模式9次＋额外文本控制6次，共15次CLSP训练 |
| CLSP预算 | pretrain300 / probe150 / fine-tune150 epochs | 每seed选256训练日；pretrain30 / probe10 / fine-tune20 |
| 选模型方式 | 固定最终epoch | 分类阶段validation probability AUC |
| 主要目的 | 在明确的本地协议下比较监督与CLSP | 可复用流程和有控制的机制/鲁棒性实验 |

独立复现修正目标梯度后，得到下表中的概率ROC-AUC。±表示20个fold/seed结果的总体标准差；由于数据划分存在重叠，这些运行并不对应20个独立人群，标准差也不具有置信区间的含义。

| 数据集 | Supervised | 修正后的Reference CLSP |
|---|---|---|
| LifeSnaps | 74.37 ± 7.06 | 73.84 ± 6.95 |
| PMData | 73.98 ± 2.49 | 71.74 ± 2.98 |

修正后的CLSP共进行了40次新训练，监督基线使用原有匹配运行的记录。在两个数据集上，CLSP均未稳定超过监督基线，因而没有重现作者报告的相对收益。预处理、context重建、split和训练选择等差异可能影响结果，但现有证据尚不足以确定其各自贡献。此前的重复行实验和test-max超参选择分析仅用于诊断，不纳入无偏最终性能的判断。

PyHealth CLSP实验使用seed0/1/2，Adam lr0.001、batch512、dropout0.1，监督分支30epochs。只有训练参与者拟合预处理并进入预训练。测试集分别为37/46/39个有标签日、9/10/9名有标签参与者，规模较小。不同seed同时改变划分和初始化，SD包含两者的变化。

## 5. 日历context延迟消融

### 研究动机

传感器记录与context更新往往具有不同的频率。模型训练时能够获得同日context，实际使用时却可能面对较旧的背景信息。我们据此提出两个假设：随着context延迟增加，压力分类AUC会下降；相比从头监督训练，CLSP的AUC下降幅度更小。该实验考察输入时效性变化下的模型行为。

![论文原图 Figure 4：作者对LifeSnaps表格特征的分析](handin_report_assets/paper_figure4_context.png)

图2来源：原论文Figure 4，PDF第8页。作者通过特征置零分析了LifeSnaps的context贡献。我们的实验进一步考察时间维度，以历史context替换当前值，研究信息更新延迟的影响。

### 实验设计

我们首先使用同日输入分别训练监督模型与CLSP，并通过验证集选择checkpoint。随后固定模型参数，在同一批测试日期上评估k=0、1、3、7日的context延迟。

对于待预测日期t，我们从同一人的历史记录中选取日期不晚于t−k的最近一条，并用其动态context替换当前日期的对应字段。年龄、性别、BMI、人格字段在本实现中视为static，保留当前值；活动/场所、心率区间分钟数等动态字段被替换。传感器、标签、测试样本和模型权重都不变，预处理也不重新拟合。

例如，在预测6月10日且k=3时，历史检索的截止日期为6月7日。若同一参与者仅在6月4日和6月9日有历史context，替换值来自6月4日。因此，k规定的是历史记录的时间上界，实际信息可能更旧。这个设计模拟了context滞后的输入条件，但没有完整模拟问卷重新采集的日程。

缺少合格历史时，实验使用训练分区的逐坐标中位数作为预先规定的fallback，替换范围仍限于动态字段。当前真实实验的fallback计数均为0；历史不足的路径通过合成数据测试验证，因此现有性能结果并未衡量真实fallback触发时的表现。

### 实验结果

以下单位均为概率ROC-AUC%，±是三seed总体SD：

| 模型 | 同日k=0 | k=1日 | k=3日 | k=7日 |
|---|---|---|---|---|
| Supervised | 52.83 ± 3.51 | 56.31 ± 2.67 | 57.47 ± 2.48 | 50.00 ± 6.04 |
| Reference CLSP | 50.03 ± 7.65 | 58.95 ± 8.08 | 51.99 ± 9.85 | 51.44 ± 10.18 |

我们的主要比较量是两种模型相对各自同日基线的AUC下降差：

```
Drop(model, k) = AUC(model, 0) - AUC(model, k)
Difference(k) = Drop(CLSP, k) - Drop(Supervised, k)
```

Drop为正表示性能下降，为负表示改善；Difference为负对应CLSP的下降幅度较小。我们使用配对区间进一步评估这一差值的不确定性。

### 配对不确定性分析

同一个人的不同日期不是互相独立的。我们以参与者为单位有放回抽样，并保留每次抽中者的全部测试日期；两模型、两个延迟条件使用完全相同的抽样，计算下降差。每个seed请求2,000次抽样，再取2.5%和97.5%分位数。seed1有14次只有一个标签类别，无法计算AUC，排除并记录；另外两个seed各2,000次有效。

| Seed | 7日CLSP减监督的下降差（百分点） | 配对bootstrap 95%区间（百分点） |
|---|---|---|
| 0 | -11.54 | [-30.94, +1.76] |
| 1 | -2.08 | [-41.29, +42.50] |
| 2 | +0.86 | [-9.85, +13.17] |

k=1、3、7的下降差区间全部包含0。性能也没有稳定地随k增加而下降。因此当前结果既不支持“context越旧一定越差”，也不建立“CLSP更抗过期”的优势；不显著也不等于两方法等效。区间基于固定模型和固定split，不包含重新训练的不确定性，三个有重叠参与者的split没有被合并成独立人群。

这一实验将同日分类性能扩展到context时效性变化的条件下。结果表明，在当前样本规模和训练预算下，CLSP对旧context的稳定性优势尚未得到支持。

## 6. 文本分支机制消融

### 研究动机

文字由已有的传感器统计量和context生成，没有引入独立测量。因此，文字对齐的潜在收益既可能来自预训练语言知识，也可能来自另一种数据表示及其对比约束。我们通过改变文字侧编码方式，检验预训练BERT在这一过程中是否提供了额外收益。

### 实验设计

| 文字侧变体 | 表示方式 | 实验控制 |
|---|---|---|
| Pretrained frozen BERT | 使用预训练语言模型表示 | BERT冻结；学习后续投影和多模态模型 |
| Random frozen BERT | 用同架构、同tokenizer，但全部权重随机初始化后冻结的BERT | 相同文字、编码维度、样本和训练协议；不是随机生成embedding向量 |
| Statistics projection | 不用BERT，直接输入55个sensor统计量＋22个context数值 | 仅训练子集拟合标准化；77维补零至768维以复用投影接口 |

三种变体都使用reference对比目标。同一个seed下保持参与者划分、选中的256个预训练日、传感器/context分支及训练预算一致。每个变体重新执行预训练→probe→fine-tune；不像日历实验只重新评估固定模型。随机BERT编码器seed固定为20261125，单独缓存，避免误读预训练BERT缓存。

### 实验结果与解释

| 预训练文字侧 | 同日概率AUC%，三seed均值±SD |
|---|---|
| 预训练冻结 BERT | 50.03 ± 7.65 |
| 随机初始化冻结 BERT | 52.50 ± 19.80 |
| 统计量直接投影 | 45.13 ± 8.82 |

预训练BERT没有稳定胜过随机冻结BERT，随机变体的波动尤其大，因此没有建立可靠的预训练语言知识优势。统计投影的均值较低，但该控制同时改变了类别表达和标准化，差异难以单独归因于语言语义。实验覆盖的小数据规模、固定预算和三个split/seed也限制了结论的普遍性，现有证据尚不足以否定BERT在其他条件下的作用。

文本机制矩阵包含3种变体×3seed，其中3个pretrained/reference运行与目标函数实验共用，另新增6次训练。

该结果使我们对文字分支的解释保持有限范围：当前实验没有分离出可靠的预训练语言知识收益，而统计投影与文字编码之间的表示差异也限制了语义层面的因果解释。

## 7. 同人和跨人AUC分解

### 研究动机

总体AUC基于高压力和低压力样本的分数排序。比较可以发生在不同人之间，也可以发生在同一个人的不同日期之间。一个模型即使主要学到了“哪些人通常分数较高”，总体AUC也可能不错，但这还不足以说明它能识别某个人压力升高的那一天。

![论文原图 Figure 8：作者报告的人口学分组与压力标签分布](handin_report_assets/paper_figure8_label_distribution.png)

图3来源：原论文Figure 8，PDF第9页。作者展示的人口学分组与标签分布差异为关注个体间差异提供了背景；我们的AUC分解进一步考察总体排序指标中同人和跨人比较的构成。

### 分解方法

我们对每个正负样本配对计分：正例预测分数更高计1，同分计0.5，否则计0。根据是否来自同一参与者，将配对分为within和between两组。两组AUC按配对数量加权后重构总体AUC：

```
Overall AUC = (Nwithin × AUCwithin + Nbetween × AUCbetween)
              / (Nwithin + Nbetween)
```

该分解按配对数加权，而非对每个人的AUC做简单平均。没有合格正负配对的组被记为未定义。计算结果通过手算例子和sklearn AUC核对。整个分析保持模型与预测不变，属于评价指标的诊断分析；bootstrap则用于量化不确定性，二者均不构成新的训练变体。

### 实验结果

| Seed | 同人正负配对数 | 跨人正负配对数 | 跨人配对占比 |
|---|---|---|---|
| 0 | 22 | 290 | 92.95% |
| 1 | 22 | 363 | 94.29% |
| 2 | 25 | 325 | 92.86% |

这些比例由测试标签和参与者构成决定，不是模型“依赖身份”的直接测量。它们说明总体指标中约93–94%的比较权重来自不同人。同日reference CLSP的同人AUC三个seed分别约54.55%、54.55%、40.00%，但每个seed仅有22–25对同人配对，估计精度有限。

总体AUC因此主要反映跨人排序，尚不足以支持可靠的个体每日追踪能力。这种指标构成的分析没有直接测量模型是否使用身份信息，也未建立人口学因素与模型行为之间的因果关系。

## 8. 输入模态与对比目标的补充控制

### MLP模态和观测偏移

我们使用现有PyHealth MLP比较了不同输入模态和Task配置，以评估sensor与context在该基线中的作用。每seed保持参与者split和样本键一致，五个配置都重新训练20epochs；以validation BCE选择checkpoint，预处理只拟合训练样本。零掉被移除的模态，保留输入维度和参数数量。

| 配置 | 概率AUC%，三seed均值±SD | 实际改变 |
|---|---|---|
| Both | 69.50 ± 3.57 | sensor和context均保留 |
| Sensors only | 68.19 ± 1.70 | context置零 |
| Context only | 55.51 ± 18.08 | sensor置零 |
| Offset 1 | 69.33 ± 0.89 | 选择同一人前1条可用观测的context，重新训练 |
| Offset 7 | 70.79 ± 2.29 | 选择同一人前7条可用观测的context，重新训练 |

共5配置×3seed＝15次训练。零输入分支可能仍有常量偏置，但不再提供该模态的样本差异信息。Both和sensor-only的均值接近，context-only波动较大。这一结果说明context单独使用时对划分较敏感，尚不足以判断其在其他模型和数据条件下的作用。

观测偏移实验按可用记录数选择历史context，并为每个配置重新训练。它考察训练任务的变化，与第5节固定权重的日历干预具有不同含义。后者检验已训练模型在context更新延迟时的行为。MLP和CLSP在split、标准化、模型结构、预算与checkpoint选择上不同，因此这里的均值不构成两类模型的直接优劣比较。

### 对比目标控制

同seed固定BERT、数据、初始化和预算，分别重新训练三种目标。Reference和stop-gradient对相同输入可有相同loss数值，但梯度通常不同；one-hot则改变目标分布。

| 目标模式 | 同日概率AUC%，三seed均值±SD |
|---|---|
| Reference soft target，保留目标梯度 | 50.03 ± 7.65 |
| Stop-gradient soft target，截断目标梯度 | 50.08 ± 12.98 |
| One-hot target，仅配对样本为正例 | 47.55 ± 9.17 |

三种目标×3seed共进行了9次训练，结果未显示稳定优胜者。Reference保留为默认模式的依据是其与参考代码目标梯度行为的一致性。该实验量化了复现过程中发现的目标梯度差异，并补充了目标分布选择的对照。

总体CLSP矩阵为9次目标训练＋6次新增文本训练＝15次；每次执行也训练匹配监督模型，重复一致的监督结果汇总时只算一次。日历重复评估不增加模型训练数。最终性能CSV为6种不同方法×4延迟×3seed＝72行。

## 9. 文档、代码质量与贡献交付

除核心实现外，我们补齐了Dataset、Task和Model的公开导出、各自的API RST文档及目录索引，并为example提供运行配置、实验设置和实测findings。Docstrings说明参数、输入输出、异常与使用方式，方法保留类型标注。[项目README](../examples/wearable/README.md)串联环境配置、组件说明、合成示例、真实训练和结果分析。这些产物对应Required File Updates以及Core Implementation中的文档和代码质量要求。

实现采用独立的数据准备、样本构造、模型计算、训练辅助和结果分析模块。贡献Python文件经过针对性的Ruff lint与format检查；Dataset的行长与命名规则检查也通过，最长行为87字符。为处理单worker环境的本地端口限制，我们在BaseDataset中接入同步事件转换路径，同时保留原多worker路径。相关共享数据和缓存行为经过回归测试，多worker执行与全量Sphinx构建尚未验证。

Ablation Study / Example Usage部分包含可运行的MLP配置实验、CLSP目标和文字控制、固定权重日历干预，以及分析入口和无身份信息的汇总结果。配置变化、数值比较和findings一起保存在贡献中，形成了从实验代码到结论的可追踪记录。MLP实验对应Dataset/Task的特征配置要求；CLSP实验提供模型机制控制，并未将其描述为学习率或隐藏维度的网格搜索。

当前代码分支包含35个相对记录基点的贡献变更文件，已经推送到个人fork。提交说明已整理作者、论文、贡献类型、文件说明和验证结果，正式上游PR尚未创建。因此，现有证据体现的是实现、文档、测试和实验成果，PR发布与外部CI/审查仍是独立的交付环节。Option 4的额外学分取决于课程对完整性与novelty的最终评定。

## 10. 测试设计与验证结果

我们通过合成数据测试检查数据处理、模型计算和分析方法的实现行为。Dataset/Task测试构造了四名虚拟参与者，包含完整天、23小时不完整天、重复小时、缺失值和无标签日。测试在真实CSV读取与事件转换路径上核验了3个监督样本、4个预训练日，以及标签、时间、输入形状和特征排除规则。

模型测试使用小型随机张量检查forward、loss、输出形状和梯度，验证三种对比目标的公式与梯度、probe阶段的参数和BatchNorm冻结，以及checkpoint恢复和Trainer单步训练。辅助测试检查参与者划分、训练集预处理、日期间隔、fallback和常量通道统计；分析测试将AUC分解与手算结果比较，将bootstrap实现与显式参与者重采样比较。共享数据加载和缓存路径也包含回归检查。

最近的相关测试套件记录为47项测试和3个子测试通过，总耗时14.82秒，另有一条依赖更新提示。最慢单项call为0.26秒，setup为0.68秒；Dataset的lint、format、命名和行宽检查也通过。全量Sphinx构建和多worker执行尚未验证。这些结果支持实现的预期行为，模型的预测效果则由前述真实数据实验评价。

## 11. 讨论与结论

这项工作的主要产出是一个可运行、经过测试的PyHealth多模态压力检测流程，以及一组围绕机制和使用条件展开的实测分析。独立复现中的CLSP没有稳定超过监督基线；受试者留出的PyHealth实验也呈现较大的跨seed波动。因此，数值上的接近或一次配置的领先，都不足以建立普遍的预训练收益。

三个核心分析进一步限定了结果的含义。日历干预没有观察到稳定的单调性能下降，也未显示CLSP更抗旧context；文本对照没有建立预训练BERT相对随机冻结编码器的稳定优势；AUC分解显示总体指标的大部分配对来自不同参与者，对个人每日变化的证据仍然有限。这些发现没有否定原方法在其他预算、划分或实现下的潜力，而是界定了当前实验能够支持的结论。

研究的限制主要来自小规模有标签测试集、有限预训练预算、三个同时改变split与初始化的seed，以及适配实现与参考流程之间的差异。Bootstrap区间反映固定模型和划分下的参与者抽样不确定性；统计投影控制还同时改变了表示方式与标准化。更大规模的受试者留出评估和更充分的训练预算仍有助于进一步检验这些机制。

## 12. Video pre 的逻辑

本项目以Option 4的Dataset + Task + Model全流程为组织主线，从压力标签稀缺和多模态输入的学习问题出发。原论文用同一天数据的数值与文字视图进行对比预训练，为利用无标签记录提供了方法。我们以Dataset完成原始数据接入，以Task定义每日输入与压力标签，以Model实现对比预训练和分类，并通过完整example连接训练与评估。文档、类型标注和测试为三个组件提供了可复用性与验证证据；复现过程中发现的目标梯度差异也完成了修正。

主要结果随后将叙述从方法实现推进到证据检验：在两数据集的独立复现以及受试者留出的PyHealth实验中，CLSP都未建立稳定优势。日历context实验进一步检验实际输入更新延迟，文字机制实验追问BERT预训练知识的贡献，AUC分解则检查总体分数是否足以反映个人的每日变化。MLP模态和对比目标控制为这些分析提供了补充背景。

我们的最终贡献由Dataset、Task、Model、完整训练example、文档测试及实测消融共同构成。它们依次展示了数据接入、任务定义、方法实现、正确性验证和科学分析。实验没有建立稳定的预训练优势，但明确了这一结论的协议范围，并对context时效性、文字机制和个体变化评价给出了进一步证据。

## 参考文献与实验材料

[项目成果与评分维度](contribution_summary.md)汇总了本报告各部分对应的已实施工作、验证结果和交付状态。

- [论文与PDF入口](https://proceedings.mlr.press/v287/yang25a.html)：作者方法、Table 1和Figure 1/4/8。原图由作者创作；本报告的中文解释由本组整理，未把原图当作本组结果。
- [PyHealth README](../examples/wearable/README.md)：环境、真实/合成运行命令与实验入口。
- [日历/文字/分解findings](../examples/wearable/lifesnaps_proposal_findings.md)、[性能72条件CSV](../examples/wearable/lifesnaps_proposal_performance.csv)、[bootstrap42记录](../examples/wearable/lifesnaps_proposal_bootstrap.csv)、[分解72记录](../examples/wearable/lifesnaps_proposal_auc_decomposition.csv)。
- [MLP完整设置与结果](../examples/wearable/lifesnaps_stressdetection_findings.md)、[目标函数对照](../examples/wearable/lifesnaps_objective_ablation.md)。
- [图像裁取来源记录](handin_report_assets/figure_sources.json)：PDF页码、图号、裁取坐标、源文件SHA256。图片只是原图区域导出，没有改变实验数值。

本报告借助Codex整理，实验数值依据保存的结果文件核对。独立repro与PyHealth example分别保存实验记录；上列代码与findings对应PyHealth贡献，论文原图保留作者归属。
