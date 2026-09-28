<!-- Unedited Markdown export of a real ResearchPilot run (free Gemini tier, arXiv only, 8 papers read in full,
     17 min, $0.00; Critic: 33 of 37 claims supported after one revision). Shown in docs/screenshots/. -->

# Recent Advancements in Hallucination Mitigation for Retrieval-Augmented Generation Systems

> **Research question:** What are recent methods for reducing hallucination in retrieval-augmented generation systems?

## Executive summary

Recent advancements in mitigating hallucinations in Large Language Models (LLMs) focus on optimizing Retrieval-Augmented Generation (RAG) through hybrid retrieval, adaptive triggering, and agentic reasoning [1, 2, 3]. Hybrid methods such as RetHyb-RRF integrate lexical and semantic retrieval to provide more relevant context, thereby reducing hallucination rates compared to standalone sparse or dense retrievers [1]. Agentic systems further enhance reliability by using modular agents to isolate hallucinations and prevent error propagation [2]. While these methods improve grounding, performance can still vary on domain-specific datasets and complex reasoning paths [1, 2].

## State-of-the-Art Mitigation Methods

Current state-of-the-art methods employ advanced retrieval architectures and iterative refinement to address factual errors. The RetHyb-RRF module utilizes WordNet for query expansion and dynamic weighting to fuse results from multiple retrieval types [1]. This approach emphasizes high-quality retrieval as a primary mechanism for providing relevant context and minimizing hallucinations in RAG systems [1]. Other frameworks, such as LLM-Augmenter, utilize Plug-and-Play modules to iteratively revise candidate responses based on feedback and external evidence chains [4].

Adaptive and agentic strategies refine the generation process by introducing dynamic intervention layers. Agentic systems, such as Multi-Agent RAG (MA-RAG), introduce multiple modular agents including a planner and evidence extractor to integrate task decomposition with evidence retrieval [2]. In multimodal domains, post-hoc RAG components perform retrieval over ad-hoc knowledge bases to assess and revise generated captions for improved faithfulness [5].

## Benchmarks and Evaluation Datasets

Standardized benchmarks are critical for quantifying hallucination rates across diverse tasks. HaluEval is a large-scale benchmark constructed to evaluate how models struggle to recognize content that conflicts with context [6]. Other core evaluation sets include HaluBench, RAGTruth, and TruthfulQA, which assess factual grounding and truthfulness [1, 2]. Domain-specific evaluation is often conducted on specialized datasets like FinanceBench, PubMed, and CovidQA, which present higher difficulty for standard retrievers [1].

For multimodal and structured output tasks, researchers utilize specialized environments. Vision-language models are evaluated using benchmarks like POPE, MSCOCO, GQA, and AMBER to validate effectiveness across diverse generative and discriminative settings [7]. In structured output tasks, RAG is employed to improve the quality of JSON workflows generated from natural language requirements [8]. These diverse benchmarks reveal that approximately 19.5% of standard ChatGPT responses were found to contain hallucinations [6].

## Comparative Performance Analysis

Comparative studies demonstrate that hybrid retrieval systems significantly outperform traditional single-method retrievers. The RetHyb-RRF retriever outperformed both sparse and dense alternatives on the HaluBenchorig dataset [1]. On the HaluEval benchmark, this hybrid approach improved accuracy to 92.00% compared to 56.00% for sparse retrievers [1]. Additionally, RAG implementation has been shown to significantly reduce hallucination and allow for better generalization to out-of-domain settings in structured output tasks [8].

Adaptive methods provide competitive accuracy while improving computational efficiency. The Rowen-Hybrid framework achieved a GPT-Judge score of 59.34 on TruthfulQA and an accuracy of 75.60 on StrategyQA, outperforming baseline consistency-based methods [3]. Rowen achieves these results with a minimal number of retrieval calls, highlighting the efficiency of adaptive retrieval [3]. In the multimodal space, faithfulness scores on ActivityNet Captions increased from 34.2% before mitigation to 92.7% after applying post-hoc RAG interventions [5].

## Open Problems and Limitations

Despite improvements, existing RAG systems face substantial challenges in domain-specific and complex reasoning contexts. Hybrid retrievers exhibit poorer performance on challenging datasets like FinanceBench, where RetHyb-RRF recorded its highest rejection rate of 32.00% [1]. Furthermore, deep reasoning chains can lead to 'overthinking,' where models dwell on irrelevant paths and generate redundant or newly hallucinated content [2]. Dynamically regulating this process requires balancing depth and efficiency to avoid hallucinations stemming from excessive reasoning [2].

Technical and architectural limitations also persist. While logit output values can help identify hallucinations, some models available through API calls do not provide them, though they are considered a supplementary rather than a necessary source for detection [4]. In multimodal scenarios, latent space steering methods like MESA must balance steering strength, as excessive steering can impair content coverage and semantic completeness [7]. Additionally, complex multi-agent mechanisms must be carefully designed to isolate hallucinations and prevent their propagation during the reasoning process [2].

## Comparison of key papers

| Ref | Paper | Year | Method | Dataset | Key result |
|---|---|---|---|---|---|
| [1] | Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis | 2025 | RetHyb-RRF (Hybrid retrieval with WordNet and RRF) | HaluBench, HaluEval, FinanceBench | MAP@3 of 0.897 on HaluBenchorig; 6% hallucination rate on HaluEval. |
| [3] | Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs | 2024 | Rowen (Adaptive RAG with consistency-based detection) | TruthfulQA, StrategyQA | 59.34 GPT-Judge score; reduced average retrieval calls to as low as 0.5. |
| [5] | ResNetVLLM-2: Addressing ResNetVLLM's Multi-Modal Hallucinations | 2025 | ResNetVLLM-2 (Faithfulness detection + post-hoc RAG) | ActivityNet Captions, ActivityNet-QA | Improved faithfulness score from 34.2% to 92.7% (97.9% in some reports). |
| [6] | HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models | 2023 | HaluEval (Sampling-then-filtering for benchmark creation) | HaluEval (35,000 samples) | Identified that 19.5% of ChatGPT responses contain hallucinations. |
| [7] | Mitigating Entangled Steering in Large Vision-Language Models for Hallucination Reduction | 2026 | MESA (Latent space steering with controlled perturbations) | POPE, MSCOCO, AMBER | 90.27% Accuracy on POPE MSCOCO; reduced CHAIR to 6.4 on AMBER. |
| [8] | Reducing hallucination in structured outputs via Retrieval-Augmented Generation | 2024 | RAG with fine-tuned siamese transformer encoder | Internal IT domain datasets (JSON workflows) | Reduced hallucination rate from 21% to <7.5% for steps and <4.5% for tables. |

## Verification

37 claims checked against the source passages: 33 supported, 1 partially supported, 3 unsupported (3 removed from the text).

## References

[1] Chandana Sree Mala, Gizem Gezici, Fosca Giannotti (2025). Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis. *arXiv (cs.IR)*. https://arxiv.org/abs/2504.05324v1
[2] Yihan Li, Xiyuan Fu, Ghanshyam Verma et al. (2025). Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems. *arXiv (cs.CL)*. https://arxiv.org/abs/2510.24476v1
[3] Hanxing Ding, Liang Pang, Zihao Wei et al. (2024). Rowen: Adaptive Retrieval-Augmented Generation for Hallucination Mitigation in LLMs. *arXiv (cs.CL)*. https://arxiv.org/abs/2402.10612v3
[4] S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain et al. (2024). A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models. *arXiv (cs.CL)*. https://arxiv.org/abs/2401.01313v3
[5] Ahmad Khalil, Mahmoud Khalil, Alioune Ngom (2025). ResNetVLLM-2: Addressing ResNetVLLM's Multi-Modal Hallucinations. *arXiv (cs.CV)*. https://arxiv.org/abs/2504.14429v1
[6] Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao et al. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models. *arXiv (cs.CL)*. https://arxiv.org/abs/2305.11747v3
[7] Yuanhong Zhang, Zhaoyang Wang, Xin Zhang et al. (2026). Mitigating Entangled Steering in Large Vision-Language Models for Hallucination Reduction. *arXiv (cs.CV)*. https://arxiv.org/abs/2604.07914v1
[8] Patrice Béchard, Orlando Marquez Ayala (2024). Reducing hallucination in structured outputs via Retrieval-Augmented Generation. *2024.naacl-industry.19*. https://arxiv.org/abs/2404.08189v1
