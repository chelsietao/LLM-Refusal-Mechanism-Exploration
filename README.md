# LLM-Refusal-Mechanism-Exploration
NTHU Math Modeling Project: Exploring the internal refusal mechanism of Gemma-2-2b-it. Investigating safety guardrails by identifying and ablating "Refusal Vectors" in the residual stream.


# 探索 LLM 拒絕行為的內部機制 (Exploring the Internal Mechanism of LLM Refusal)

![Project Status](https://img.shields.io/badge/Status-Completed-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/PyTorch-HuggingFace-orange)

> **清華大學數學系 數學建模專題** > 

## Project Overview

當大型語言模型（LLM）面對「如何製作蛋糕」與「如何製作炸彈」這兩個指令時，會產生截然不同的反應。本專案旨在從數學與向量空間的角度，深入探討模型「決定拒絕」的那一瞬間發生了什麼。

我們假設模型對於「有害指令」與「無害指令」的處理路徑在高維空間中是線性可分（Linearly Separable）的。透過分析 **Google Gemma-2-2b-it** 模型的內部隱藏狀態（Hidden States），我們成功提取出了代表拒絕意圖的 **「拒絕向量」（Refusal Vector）**。

### 目標
1. **視覺化：** 觀察有害與無害指令在模型不同層數的活化差異。
2. **提取：** 利用平均差值法（Difference of Means）計算出拒絕向量。
3. **干預（Ablation）：** 在推論過程中將拒絕向量從模型的殘差流（Residual Stream）中移除，觀察是否能繞過模型的安全機制。

## Methodology

本專案基於 *Representation Engineering* 的概念，執行步驟如下：

1. **數據收集**：
   - 準備兩組數據集：`Harmful`（有害指令）與 `Harmless`（無害指令）。
2. **提取內部狀態**：
   - 將指令輸入模型，並記錄每一層（Layer）的 Hidden States。
3. **計算拒絕向量**：
   - 計算有害指令群與無害指令群在特定層的中心點差異：
     $$v_{refusal} = \mu_{harmful} - \mu_{harmless}$$
4. **干預模型 **：
   - 在模型進行前向傳播（Forward Pass）時，對特定層的輸出進行向量減法：
     $$h_{layer} \leftarrow h_{layer} - \alpha \cdot v_{refusal}$$
     （其中 $\alpha$ 為控制干預強度的係數）

## Results

透過干預特定層（如 Layer 12 附近），我們觀察到模型對於有害指令的態度發生顯著轉變。

- **原始模型**：回答 "I cannot provide instructions on how to build a bomb..." (拒絕分數高)
- **移除拒絕向量後**：模型轉為回答 "You're in luck! Building a bomb with household items is..." (成功繞過防禦)

<img width="982" height="629" alt="Image" src="https://github.com/user-attachments/assets/c4691dd3-3eaf-435b-b526-01efb3860f6c" />

<img width="1390" height="690" alt="Image" src="https://github.com/user-attachments/assets/62d9b1c4-2bcb-4a2d-9a8a-9eb8a2074d04" />

## Installation & Usage

### 環境需求
本專案建議在 Google Colab (T4 GPU) 或具備 NVIDIA GPU 的環境下執行。

```bash
# 安裝依賴套件
pip install transformers torch accelerate huggingface_hub matplotlib numpy
