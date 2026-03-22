# STAT 340 Lecture 14：简单线性回归（续）& 多元线性回归入门

> **课程**：STAT 340 Data Science Modeling II  
> **讲师**：Yongyi Guo  
> **内容来源**：课件 L15.pdf + 课堂录音转录  
> **考试提示**：Midterm 2 覆盖范围从假设检验第二部分到简单线性回归（不含多元线性回归）

---

## 一、简单线性回归 (Simple Linear Regression) 回顾

### 1.1 模型定义

简单线性回归建模 $Y$ 在给定 $X$ 条件下的分布：

$$Y_i = \beta_0 + \beta_1 x_i + \epsilon_i, \quad i = 1, \dots, n$$

- $\beta_0$（截距）、$\beta_1$（斜率）：未知回归系数
- $\epsilon_i$：噪声项，独立同分布，均值为 0，方差为 $\sigma^2$
- 更强假设下：$\epsilon_i \sim N(0, \sigma^2)$
- $x_i$ 视为固定（非随机）或条件给定

**三个模型参数**：$\beta_0$、$\beta_1$、$\sigma^2$

**老师补充**：我们只建模 $\epsilon_i$ 的随机性，不关心 $x_i$ 是否随机。给定 $x_i$，$Y_i$ 的条件均值为线性函数 $\beta_0 + \beta_1 x_i$，条件方差为 $\sigma^2$。$\sigma^2$ 描述数据集的噪声水平——噪声大则难以精确预测，噪声小则数据可预测性强。

### 1.2 R 中运行简单线性回归

以亚特兰大大气铅含量预测 22 年后的严重攻击犯罪率为例：

```r
atlanta_lead_lm <- lm(aggr.assault.per.million ~ 1 + air.pb.metric.tons,
                      data = atlanta_lead)
summary(atlanta_lead_lm)
```

输出关键信息：
- **截距** $\hat{\beta}_0 = 107.94$，标准误 80.46，p-value = 0.189（不显著）
- **斜率** $\hat{\beta}_1 = 1.40$，标准误 0.08，p-value < 2e-16（极显著 `***`）
- **Residual standard error**: 180.6，df = 34
- **R² = 0.898**，Adjusted R² = 0.895
- **F-statistic**: 299.4，p-value < 2.2e-16

---

## 二、回归系数的置信区间与假设检验

### 2.1 置信区间的计算

对 $\beta_j$ 的 95% 置信区间：

$$\hat{\beta}_j \pm t_{0.975, \, n-2} \cdot \text{SE}(\hat{\beta}_j) \approx \hat{\beta}_j \pm 2 \cdot \text{SE}(\hat{\beta}_j)$$

R 中使用 `confint()` 直接获取：

```r
confint(atlanta_lead_lm, level = 0.95)
```

输出：
- 截距 $\beta_0$：$[-55.58, \; 271.47]$（包含 0 → 不显著）
- 斜率 $\beta_1$：$[1.24, \; 1.57]$（不包含 0 → 显著）

**老师详解**：置信区间的两端就是 $\hat{\beta}_j \pm t_{\text{quantile}} \times \text{SE}(\hat{\beta}_j)$。R 用精确的 t 分位数计算，手算时近似用 2 即可。例如 $\beta_0$ 的 CI ≈ $107.94 \pm 2 \times 80.46$。

### 2.2 置信区间与假设检验的对偶性

- **变量显著** ⟺ **p-value 很小** ⟺ **95% CI 不包含 0**
- 从表中可看出：截距的 CI 包含 0（p = 0.189，不拒绝 $H_0: \beta_0 = 0$），斜率的 CI 不包含 0（p < 2e-16，拒绝 $H_0: \beta_1 = 0$）

**老师强调**：这两种不确定性量化方法是等价的。用同样的 estimate 和 SE，既能算 t 统计量得 p-value，也能构造 CI。若 95% CI 覆盖 0 则在 5% 水平不拒绝，反之拒绝。

---

## 三、预测 (Making Predictions)

### 3.1 基本方法

给定新的 $x_{\text{new}}$，预测值为：

$$\hat{y}_{\text{new}} = \hat{\beta}_0 + \hat{\beta}_1 \cdot x_{\text{new}}$$

```r
predict(atlanta_lead_lm, newdata = data.frame(air.pb.metric.tons = 1300))
## 1932.813
```

**老师解释**：拟合线之外的部分是我们永远无法观测的独立噪声，所以最佳预测就是把新 $x$ 代入拟合的线性函数。

### 3.2 外推的危险 (Extrapolation)

如果新的 $x$ 值远超训练数据范围，预测不可靠。

**例子**：若铅含量为 2000 公吨（数据范围约 400–1500），代入模型预测没有意义。因为线性关系通常只在观测范围内近似成立。

**老师举的另一个例子**："sustainable" 一词在英语中的使用频率随年份线性增长，但如果外推到未来，频率会超过 100%，这显然荒谬。

---

## 四、模型拟合评估 (Assessing Model Fit)

### 4.1 残差平方和 (RSS / SSE)

模型通过最小化平方和来拟合：

$$\ell(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i))^2$$

**RSS（残差平方和）**：

$$\text{RSS} = \text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

RSS 越小，模型拟合越好。但我们需要一个标准来判断"多小算小"。

### 4.2 R²（决定系数）

**总平方和 (TSS)**：将所有 $y_i$ 用 $\bar{y}$ 预测时的残差平方和（"最笨模型"的表现）

$$\text{TSS} = \sum_{i=1}^{n} (y_i - \bar{y})^2$$

**R² 定义**：

$$R^2 = \frac{\text{TSS} - \text{RSS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}}$$

- $R^2 \in [0, 1]$
- $R^2$ 接近 1 → 模型解释了大部分变异 → 好的拟合
- $R^2$ 接近 0 → 模型几乎没有比"用 $\bar{y}$ 预测一切"好多少

**老师直觉解释**：
- TSS 是"最笨模型"（只用 $\bar{y}$ 预测一切）的残差平方和
- RSS 是我们的模型的残差平方和，RSS ≤ TSS（因为更灵活的直线不可能比水平线更差）
- $R^2$ 衡量从"最笨模型"到"我们的模型"，残差减少了多少比例
- 在简单线性回归中，$R^2 = r^2$（样本相关系数的平方）

**亚特兰大例子中** $R^2 = 0.898$，说明铅含量解释了约 89.8% 的攻击率变异，拟合很好。

### 4.3 F 统计量与整体模型显著性

**零假设**：$H_0: \beta_1 = \beta_2 = \dots = \beta_p = 0$（除截距外所有回归系数为 0）

**F 统计量**：

$$F = \frac{\text{MSS}/p}{\text{RSS}/(n-p-1)} = \frac{\frac{1}{p}\sum(\hat{y}_i - \bar{y})^2}{\frac{1}{n-p-1}\sum(y_i - \hat{y}_i)^2}$$

其中 MSS（模型平方和）$= \sum(\hat{y}_i - \bar{y})^2$，且 TSS = RSS + MSS。

在 $H_0$ 下且 $\epsilon_i \sim \text{iid } N(0, \sigma^2)$ 时，$F \sim F(p, \, n-p-1)$。

**老师补充**：
- **t 分布**来源于"标准正态 / 估计的标准差"的比值
- **F 分布**来源于两个方差估计的比值
- 在简单线性回归中，F 检验和 $\beta_1$ 的 t 检验实质等价（零假设相同，p-value 也相同）
- 在多元线性回归中，二者不同：t 检验针对单个 $\beta_j = 0$，F 检验针对所有 $\beta_j = 0$

### 4.4 残差标准误 (RSE)

$$\text{RSE} = \sqrt{\frac{\text{RSS}}{n - (p+1)}} = \sqrt{\frac{\sum(y_i - \hat{y}_i)^2}{n - (p+1)}}$$

- 自由度 df = 数据点数 − 参数数
- RSS 的单位是平方单位，取平方根后 RSE 与 $Y$ 同单位
- 参数越多 → 分母越小 → RSE 会受影响

---

## 五、练习题解析

### 练习题 13：诊断图与 SLM 假设违反

给定 fitted vs residuals 图和 QQ 图，问是否违反 SLM 假设。

**答案：C — 残差 vs 拟合值图显示违反了同方差性 (homoscedasticity) 假设**

**老师解析**：
- QQ 图检查**正态性**假设（残差 vs 标准正态分位数），不是线性假设 → A 错
- Fitted vs residual 图不直接检查**独立性** → B 错
- 图中小拟合值对应残差几乎不变化，大拟合值对应残差变化很大 → 方差不恒定 → C 正确
- D 明显错误，诊断图正是为检查假设而画的

### 练习题 17：置信区间与历史数据一致性

公园历史观察：温度升1度 → 游客增加8人。模型 $\hat{\beta}_1 = 7.33$，SE = 0.3881。

问：分析数据是否与历史一致？

**答案：C — 是的，因为 8 在 95% CI [6.55, 8.11] 内**

**老师解析**：
- 95% CI ≈ $7.33 \pm 2 \times 0.3881 \approx [6.55, 8.11]$
- 8 在此区间内 → 数据与历史观察一致
- p-value 对应的零假设是 $\beta_1 = 0$，不是 $\beta_1 = 8$，所以不能用 p-value 来回答"是否与 8 一致"

### 练习题 21：温度的效果是否显著

**答案：C — 是的，因为 p-value 很小**

**老师解析**：
- "温度对游客有显著效果" ⟺ 拒绝 $H_0: \beta_1 = 0$
- 对应 Temperature 行的 p-value < 2e-16，极小 → 显著拒绝
- 判断显著性直接看 p-value，不是看系数大小或标准误大小

---

## 六、多元线性回归 (Multiple Linear Regression) 入门

> **注意**：此部分不在 Midterm 2 考试范围内

### 6.1 动机

单一预测变量往往不够：
- 预测工资：不仅看教育年限，还看专业、人口统计信息、父母教育水平等
- 预测房价：需要面积、房龄、卧室数、周边设施等多个因素

### 6.2 模型形式

对每个观测 $i = 1, \dots, n$：

$$Y_i = \beta_0 + \beta_1 X_{i,1} + \beta_2 X_{i,2} + \dots + \beta_p X_{i,p} + \epsilon_i$$

- $X_i = (X_{i,1}, \dots, X_{i,p})^\top \in \mathbb{R}^p$：第 $i$ 个观测的 $p$ 个预测变量
- $\epsilon_i$：独立噪声

### 6.3 矩阵表示

定义**设计矩阵**（design matrix）：

$$\mathbf{X} = \begin{bmatrix} 1 & X_{1,1} & \cdots & X_{1,p} \\ 1 & X_{2,1} & \cdots & X_{2,p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & X_{n,1} & \cdots & X_{n,p} \end{bmatrix}$$

紧凑表示：$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$

- $\mathbf{y} = (y_1, \dots, y_n)^\top$：响应向量
- $\boldsymbol{\beta} = (\beta_0, \beta_1, \dots, \beta_p)^\top$：回归系数向量（固定未知）
- $\boldsymbol{\epsilon} = (\epsilon_1, \dots, \epsilon_n)^\top$：噪声向量（唯一随机源）
  - 若 $\epsilon_i \sim N(0, \sigma^2)$，则 $\boldsymbol{\epsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I}_n)$

**老师补充**："设计矩阵"名字来源于农业实验中研究者可以"设计"（控制）$X$ 的取值。但在回归分析中，不管 $X$ 怎么来的，我们都将其视为固定/条件给定。

### 6.4 mtcars 数据集实例

用 disp（排量）、hp（马力）、wt（重量）预测 qsec（四分之一英里时间）：

```r
mtc_model <- lm(qsec ~ 1 + disp + hp + wt, data = mtcars)
summary(mtc_model)
```

输出：
| 变量 | Estimate | Std. Error | t value | Pr(>\|t\|) |
|---|---|---|---|---|
| (Intercept) | 17.965 | 0.850 | 21.144 | < 2e-16 *** |
| disp | −0.0066 | 0.0042 | −1.590 | 0.123 |
| hp | −0.0230 | 0.0046 | −4.986 | 2.88e-05 *** |
| wt | 1.485 | 0.429 | 3.461 | 0.00175 ** |

- RSE = 1.062，df = 28（n=32，4个参数）
- R² = 0.6808，Adjusted R² = 0.6466
- F = 19.91，p-value = 4.134e-07

### 6.5 系数解释

$\hat{\beta}_3 = 1.485$（wt 的系数）：**在其他变量保持不变的条件下**，重量每增加 1 单位（1000磅），四分之一英里时间平均增加约 1.5 秒。

**显著性判断**：
- Intercept、hp、wt 显著（p-value < 0.05）
- disp 不显著（p = 0.123）→ 在其他变量已存在的情况下，disp 的额外贡献不显著

### 6.6 残差诊断

QQ 图显示残差尾部偏重（如 Merc 230 等车型为异常值）。

**老师建议**：
- QQ 图若不理想，不代表不能用模型，但应对依赖正态性的推断（如 p-value）持谨慎态度
- 实际中遇到异常点应回到数据检查具体观测，可能是录入错误或特殊车型
- 即使残差不正态，当 $n \to \infty$ 时推断渐近有效

### 6.7 多元回归的模型拟合评估

- **RSE 的自由度**：$\text{df} = n - (p+1)$，参数越多分母越小
- **R²**：定义同简单回归，但多元回归中 R² 会随预测变量增加而单调递增（即使新变量无用）→ 需要用 **Adjusted R²** 修正
- **F 统计量**：检验 $H_0: \beta_1 = \dots = \beta_p = 0$（所有预测变量同时无用）
  - 在多元回归中，F 检验与单个 t 检验含义不同：t 检验只针对单个 $\beta_j = 0$

---

## 七、核心概念速查

| 概念 | 公式/定义 | 直觉理解 |
|---|---|---|
| RSS | $\sum(y_i - \hat{y}_i)^2$ | 模型未解释的变异 |
| TSS | $\sum(y_i - \bar{y})^2$ | 数据总变异（最笨模型的 RSS） |
| MSS (ESS) | $\sum(\hat{y}_i - \bar{y})^2$ | 模型解释的变异，TSS = RSS + MSS |
| R² | $1 - \text{RSS}/\text{TSS}$ | 模型解释的变异比例，越接近 1 越好 |
| RSE | $\sqrt{\text{RSS}/\text{df}}$ | 残差的标准差估计，与 Y 同单位 |
| F 统计量 | $(\text{MSS}/p) / (\text{RSS}/(n-p-1))$ | 模型方差 vs 噪声方差的比值 |
| CI for $\beta_j$ | $\hat{\beta}_j \pm t_{0.975,\text{df}} \cdot \text{SE}(\hat{\beta}_j)$ | 手算近似用 $\pm 2 \cdot \text{SE}$ |
