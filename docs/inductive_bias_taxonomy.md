# Architecture taxonomy by inductive bias

This restores the full user-supplied taxonomy. It is a creative search library, not a queue of models to train. The machine-readable names are in [`inductive_bias_taxonomy.json`](../research/inductive_bias_taxonomy.json).

## Levels of assumption

| Level | Assumption added | Families retained |
|---:|---|---|
| 0 | the datum is a vector/distribution | feed-forward, autoencoders, energy/equilibrium, reservoir, probabilistic |
| 1 | generic relation, symmetry, metric, memory or state-space structure | equivariant models, attention, metric learning, external memory, SSMs |
| 2 | order, causality and/or physical timing matter | discrete/continuous recurrence, event models, sequence attention, time encoding |
| 3 | locality, translation, spatial hierarchy or resolution bands matter | convolutions, CNN families, pyramids, wavelet/multiresolution |
| 4 | an explicit graph, hierarchy, combinatorial structure or geometry exists | GNNs, higher-order graphs, trees/DAGs, hyperbolic and relational models |

### Level 0 — universal foundations

- Feed-forward: MLP, RBF, KAN, MLP-Mixer, gMLP, ResMLP.
- Reconstruction: vanilla/denoising/sparse/contractive AE, VAE variants, VQ-VAE, masked AE.
- Energy/equilibrium: Hopfield, RBM/DBN/DBM, Helmholtz machine, EBM.
- Reservoir: Echo State Network, Liquid State Machine.
- Probabilistic: Bayesian networks, Deep Gaussian Processes, Neural Processes.

### Level 1 — weak structural bias

- Equivariance: G-CNN, steerable, E(n)/SE(3), tensor-field, Lie/gauge, capsules.
- Generic attention: Transformer and efficient variants, Perceiver, Set Transformer, Slot Attention.
- Metric learning: Siamese, triplet, prototypical and contrastive patterns.
- External memory: NTM, DNC and memory networks.
- State-space: S4/S4D/S5, Mamba, H3, Hyena, RWKV, RetNet, Griffin, xLSTM.

### Level 2 — temporal bias

- Discrete recurrence: RNN, LSTM, GRU, Clockwork/hierarchical multiscale, unitary/orthogonal, QRNN/SRU.
- Continuous time: Neural ODE/CDE/SDE, LTC, CTRNN, Latent ODE and ODE-RNN.
- Event-driven: point processes, Neural Hawkes and asynchronous timestamp models.
- Sequence attention: seq2seq, autoregressive/bidirectional/encoder-decoder, Transformer-XL, Universal Transformer, ACT.
- Explicit timing: sinusoidal/learned/relative positions, RoPE, ALiBi, CoPE and T5 bias.

### Level 3 — spatial and multiresolution bias

- Atomic convolutions: standard, depthwise/separable/grouped, dilated, deformable, transposed, 1×1, 3D/(2+1)D, dynamic/conditional.
- CNN lineages: LeNet through ResNet/ResNeXt/DenseNet; SE, CBAM, ECA.
- Efficient/modern: MobileNet, ShuffleNet, EfficientNet, ConvNeXt, RepVGG, large-kernel, FocalNet, HorNet.
- Conv/attention hybrids: ViT lineage, Swin, CoAtNet, MaxViT, CvT, MetaFormer/PoolFormer.
- Explicit scales: FPN/PANet/BiFPN/NAS-FPN/ASPP, scattering and wavelet networks.

### Level 4 — relational and topological bias

- Message passing: GCN, GraphSAGE, GAT, GIN, MPNN, spectral graph convolution and pooling.
- Higher structure: temporal graphs, hypergraphs, simplicial/cellular/sheaf networks.
- Combinatorial: recursive networks, Tree-LSTM/GRU, DAG and algebraic structures.
- Non-Euclidean: hyperbolic, Poincaré, Lorentzian and mixed-curvature spaces.
- Explicit relation: Relation/Interaction Networks, Neural Relational Inference and Relational Memory.

## Cross-cutting primitives

Residual/dense paths, normalization, gating, MoE, hypernetworks, dropout and stochastic depth are mechanisms, not complete architecture families. They are screened through measured functions such as information flow, conditioning, regime selection or regularization.

## How it enters the Hay program

The taxonomy generates diversity. The current catalog contains only architectures with an active Hay hypothesis. A new family enters the active screen when a diagnostic or capability task names what its bias should solve. Thus KAN, attention, wavelets or hyperbolic models are remembered without being scheduled arbitrarily.

