Here's a **senior-level roadmap** for mastering LLMs — not just toy examples, but the depth needed to build, fine-tune, and deploy production-grade systems.

---

## Phase 0: Prerequisites (2–4 weeks)
**Don't skip this.** Most people crash here because they try to run before they can walk.

| Topic | What to Actually Know |
|-------|----------------------|
| **Linear Algebra** | Matrix ops, eigenvalues, SVD, gradients. *Focus:* How weight matrices transform vectors. |
| **Probability & Stats** | MLE, KL divergence, entropy, Bayesian basics. *Focus:* Why cross-entropy loss works. |
| **Calculus** | Chain rule, backprop intuition, partial derivatives. |
| **Python + PyTorch** | Tensors, autograd, `nn.Module`, data loaders. Build a small MLP from scratch. |
| **ML Basics** | Logistic regression, embeddings, overfitting, train/val/test splits. |

**Project:** Train a character-level RNN (Karpathy-style) on a small text corpus. Understand why it fails to capture long-range dependencies.

---

## Phase 1: The Transformer Architecture (3–4 weeks)
This is the foundation. Everything else is a variation.

### Core Papers (Read in this order)
1. **"Attention Is All You Need"** (Vaswani et al., 2017) — Read the original. Implement multi-head attention from scratch.
2. **"The Illustrated Transformer"** (Jay Alammar) — Visual intuition before code.
3. **"BERT: Pre-training of Deep Bidirectional Transformers"** — Understand MLM and NSP.
4. **"Language Models are Unsupervised Multitask Learners"** (GPT-2) — Understand autoregressive pretraining.

### What to Build
- **From-scratch Transformer:** Implement encoder + decoder with pure PyTorch (no `nn.Transformer`). Train on a tiny translation or summarization task.
- **Understand:**
  - Positional encodings (sinusoidal vs. learned vs. RoPE)
  - LayerNorm placement (Pre-LN vs. Post-LN)
  - Residual connections and why they matter for 100+ layer models
  - Causal masking in decoders

---

## Phase 2: Modern LLM Architectures (2–3 weeks)
Move beyond the original transformer to what actually powers GPT-4, Claude, Llama, etc.

| Innovation | Why It Matters |
|------------|---------------|
| **RoPE (Rotary Position Embedding)** | Relative positions, better long-context extrapolation. |
| **SwiGLU / Gated MLP** | Better activation than ReLU/GELU. |
| **RMSNorm** | Simpler, more stable than LayerNorm. |
| **Grouped Query Attention (GQA)** | Reduces KV cache memory by 4-8x. Critical for inference. |
| **Sliding Window Attention** | Long context without quadratic cost (Mistral). |

**Read:**
- **Llama 2 paper** — The modern "reference architecture"
- **Mistral 7B paper** — Sliding window + GQA
- **Chinchilla paper** — Compute-optimal scaling laws (model size vs. data)

**Project:** Re-implement a Llama-style decoder-only model. Load actual Llama-2-7B weights and verify your implementation produces identical logits.

---

## Phase 3: Training at Scale (4–6 weeks)
This separates hobbyists from engineers.

### 3A: Pre-training
- **Data pipeline:** Deduplication (MinHash), quality filtering (perplexity scoring), tokenization (BPE/SentencePiece — build your own tokenizer with `tokenizers`).
- **Distributed training:** Data parallelism, tensor parallelism, pipeline parallelism, ZeRO (DeepSpeed), FSDP.
- **Mixed precision:** FP16/BF16, gradient scaling.
- **Stability tricks:** Gradient clipping, learning rate schedules (cosine with warmup), loss spikes debugging.

**Resource:** [LLM Training Handbook](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) + DeepSpeed docs.

### 3B: Fine-tuning
- **Full fine-tuning:** When and why it's still used.
- **PEFT methods:**
  - **LoRA / QLoRA:** Understand low-rank adaptation mathematically. Why rank=16 works.
  - **Prefix tuning / Prompt tuning:** Soft prompt embeddings.
- **Instruction tuning:** Format data as `{instruction} + {input} → {output}`. Understand chat templates (`<|user|>`, `<|assistant|>`).
- **Multi-turn conversation formatting.**

**Project:** Fine-tune Llama-3-8B on a custom dataset using QLoRA (4-bit). Then do full fine-tuning on a single A100 with DeepSpeed ZeRO-3.

---

## Phase 4: Alignment & Post-training (3–4 weeks)
This is how "raw" models become ChatGPT.

| Technique | Core Idea |
|-----------|-----------|
| **SFT (Supervised Fine-Tuning)** | Teach the model to follow instructions with high-quality demonstrations. |
| **RLHF** | Train a reward model on human preferences, then optimize policy with PPO. Understand the KL penalty. |
| **DPO (Direct Preference Optimization)** | Skip the reward model — optimize directly on preference pairs. Simpler, often better. |
| **KTO / IPO** | Newer alignment methods that don't need pairs. |

**Papers:**
- **InstructGPT** — RLHF origin
- **DPO paper** (Rafailov et al., 2023)
- **Constitutional AI** (Anthropic) — Self-improvement without human labels

**Project:** Implement DPO from scratch. Collect ~1K preference pairs (or use Anthropic HH-RLHF). Align a 1B model.

---

## Phase 5: Inference Optimization (3–4 weeks)
**Latency and cost matter more than benchmark scores in production.**

| Technique | What It Does |
|-----------|-------------|
| **KV Cache** | Avoid recomputing key/values for prior tokens. Understand memory growth: `2 * n_layers * n_heads * d_head * seq_len * batch * bytes_per_param` |
| **Quantization** | INT8, INT4, GPTQ, AWQ, GGUF. Understand asymmetric vs. symmetric quantization, grouping. |
| **Continuous Batching** | vLLM / TGI — batch requests dynamically. |
| **PagedAttention** | vLLM's key innovation — non-contiguous KV cache memory. |
| **Speculative Decoding** | Draft small model → verify with large model. 2-3x speedup. |
| **FlashAttention** | IO-aware exact attention. Re-implement the algorithm. |

**Tools to Master:**
- **vLLM** — Production serving
- **TensorRT-LLM** — NVIDIA optimized inference
- **llama.cpp** — Edge/CPU deployment

**Project:** Deploy a 70B model on a single 48GB GPU using 4-bit quantization + vLLM. Benchmark throughput (tok/sec) vs. latency.

---

## Phase 6: RAG, Agents & Tool Use (3–4 weeks)
LLMs don't exist in isolation.

### RAG (Retrieval-Augmented Generation)
- **Embedding models:** E5, BGE, GTE — not just "OpenAI embeddings"
- **Vector DBs:** FAISS, Milvus, pgvector — understand HNSW indexing, not just API calls
- **Chunking strategies:** Semantic chunking, recursive splitting, late chunking
- **Reranking:** Cross-encoder rerankers (ColBERT, BGE-reranker)
- **Advanced RAG:** Query expansion, HyDE, self-RAG, corrective RAG

### Agents & Tool Use
- **ReAct pattern:** Reasoning + Acting loop
- **Function calling:** Schema-constrained generation (JSON mode, constrained decoding)
- **Multi-agent systems:** AutoGen, CrewAI patterns
- **Planning:** Tree of Thoughts, Graph of Thoughts

**Project:** Build a research assistant that: (1) searches arXiv via API, (2) retrieves relevant papers, (3) synthesizes answers with citations, (4) can generate and execute Python code to verify claims.

---

## Phase 7: Evaluation & Safety (2–3 weeks)
*"If you can't measure it, you can't improve it."*

### Evaluation
- **Perplexity** — Poor proxy for usefulness, good for training health
- **Benchmarks:** MMLU, GSM8K (math), HumanEval (code), MT-Bench (chat), Arena ELO
- **Custom evals:** Build task-specific evals. LLM-as-a-judge (with proper prompt engineering and calibration).
- **Red-teaming:** Adversarial prompts, jailbreaks, bias testing

### Safety
- **Prompt injection:** Direct vs. indirect. Mitigations (input/output filtering, privilege separation).
- **Hallucination detection:** RAG grounding, self-consistency checking.
- **Alignment faking / deception:** Frontier research topic.

---

## Phase 8: Research Frontiers (Ongoing)
Stay current. Subscribe to [Papers With Code](https://paperswithcode.com/) and [AK's Twitter](https://twitter.com/_akhaliq).

| Area | Key Developments |
|------|-----------------|
| **Mixture of Experts (MoE)** | Sparse activation — Mixtral 8x7B, DeepSeek-V2 |
| **Long Context** | Ring attention, linear attention (Mamba, RWKV), 1M+ token contexts |
| **Multimodal** | LLaVA, GPT-4V — vision + text alignment |
| **Reasoning Models** | o1-style chain-of-thought scaling, process reward models, test-time compute |
| **Test-Time Training** | Adapt model weights at inference time |
| **Efficient Architectures** | Mamba (SSMs), Griffin (Gated Linear RNNs) — alternatives to attention |

---

## Suggested Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Prerequisites | 3 weeks | 3 weeks |
| Transformers | 4 weeks | 7 weeks |
| Modern Architectures | 3 weeks | 10 weeks |
| Training | 6 weeks | 16 weeks |
| Alignment | 4 weeks | 20 weeks |
| Inference | 4 weeks | 24 weeks |
| RAG/Agents | 4 weeks | 28 weeks |
| Evaluation | 3 weeks | 31 weeks |

**~7–8 months** of focused, deep work to reach senior-level competence.

---

## Resources That Actually Matter

**Courses:**
- **CS224N** (Stanford NLP) — Foundations
- **CS324** (Stanford Large Language Models) — Advanced
- **Full Stack LLM Bootcamp** — Applied engineering

**Codebases to Study:**
- **nanoGPT** (Karpathy) — Clean transformer training
- **llama.cpp** — Inference optimization
- **vLLM** — Production serving
- **trl** (HuggingFace) — RLHF/DPO implementations
- **unsloth** — Memory-efficient fine-tuning

**Key Papers (The "Canon"):**
1. Attention Is All You Need
2. BERT / GPT-2 / GPT-3
3. Training Language Models to Follow Instructions (InstructGPT)
4. Llama 2 / Llama 3
5. DPO
6. LoRA
7. FlashAttention
8. vLLM (PagedAttention)

---

## Final Advice from a Senior Perspective

1. **Build from scratch first, then use libraries.** Anyone can call `model.generate()`. Few can debug why loss spikes at layer 47.
2. **Read papers with code.** Reproduce the key result before moving on.
3. **Focus on one scale.** Understand 1B models deeply before worrying about 100B+.
4. **Engineering > Math for most roles.** Distributed systems, CUDA kernels, and data pipelines often matter more than novel architectures.
5. **Start a blog.** Explain what you learn. If you can't explain it simply, you don't understand it.

The field moves fast, but fundamentals are timeless. Master the transformer, understand scaling laws, and build real systems. The rest is iteration.
