# FedFree: A Data-Free Model Poisoning Attack Against Federated Learning

Official implementation of the paper:
FedFree: A Data-Free Model Poisoning Attack Against Federated Learning
by Zhenquan Qin, Kehao A, Bingxian Lu, and Guangjie Han

## 🧠 Overview

Federated Learning (FL) enables distributed training without sharing raw data, but remains vulnerable to model poisoning attacks.
FedFree introduces a data-free poisoning framework that operates entirely without client data, leveraging only the server-broadcast global parameters.

The framework employs a block-wise Shampoo-inspired parameter generator to synthesize benign-like updates and a multi-perturbation adversarial mechanism to produce statistically consistent yet malicious updates.

## 🚀 Key Features

Data-Free Operation: Requires no access to any client-side data.

Block-Wise Parameter Synthesizer: Generates realistic benign-like parameters using only global model history.

Adaptive Norm Clipping: Maintains stealth by constraining perturbations within benign statistical envelopes.

Robust Effectiveness: Outperforms existing data-dependent and data-free attacks under multiple robust aggregation defenses.

## 🧩 Experimental Setup

Datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100

Defenses Tested: FedAvg, Multi-Krum, Trimmed-Mean, Bulyan, CC, RFA, RoFL

Metrics: Attack Impact, ∆AUC@100, cosine similarity, norm ratio
