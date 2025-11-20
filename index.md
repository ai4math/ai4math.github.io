

The website AI for Mathematics (AI4Math) that is under construction.

- [Optimization: Stochastic Gradient Descent](optimization-1/)


# AI for Mathematics

The field of **"AI for Mathematics" (AI4Math)** has evolved rapidly from traditional symbolic logic to modern neuro-symbolic approaches. While early artificial intelligence focused on rigid, rule-based systems (GOFAI), the current wave of research integrates Deep Learning (DL) and Large Language Models (LLMs) with formal verification systems like *Lean*, *Coq*, and *Isabelle*. 

The core challenge in this domain is the "formal-informal gap": bridging the intuitive, flexible reasoning of human mathematics (informal) with the rigorous, machine-checkable guarantees of formal proof assistants.

The literature provided outlines a clear trajectory: moving from Reinforcement Learning (RL) applied to proof search, to Generative AI that conjectures theorems and proofs, and finally to "human-in-the-loop" systems that aid mathematical discovery.\\




## Automated Theorem Proving (ATP)
The central pillar of AI4Math is Automated Theorem Proving—using machines to generate valid proofs within a formal system.

### From RL to Generative Models
Early neural approaches attempted to guide search algorithms using deep networks. 
* **Holophrasm (2016) and DeepMath (2016):** Pioneering works that used neural networks to select premises or guide proof search in higher-order logic.

As Reinforcement Learning matured, it was applied to navigate the infinite search space of mathematical tactics:
* **TacticZero (2021):** Demonstrated that an agent could learn to prove theorems from scratch without human supervision by interacting with proof assistants.
* **HOList (2019):** Provided a robust environment for machine learning in higher-order logic, establishing a benchmark for RL agents.

However, the advent of Transformers shifted the paradigm towards generative language modeling. Instead of just selecting from a pre-defined list of tactics, models began generating the tactics themselves:
* **Generative Language Modeling for ATP (2020):** Polu and Sutskever showed that language models could predict proof steps effectively.
* **Thor (2022) and Baldur (2023):** Integrated LLMs with automated provers (hammers), allowing the model to wield external tools to discharge sub-goals.
* **DeepSeek-Prover (2024):** Pushed this further by leveraging large-scale synthetic data to train LLMs specifically for formal proving.

### Retrieval-Augmented Proving
A major limitation of pure LLMs is their inability to access the vast library of existing mathematical lemmas during inference. 
* **LeanDojo (2023):** Addressed this by introducing retrieval-augmented generation to theorem proving, allowing the model to dynamically retrieve relevant premises from the math library before generating a proof step.

### Advances in Geometry Problems
Geometry has proven to be a specialized domain where neuro-symbolic AI excels. Recent approaches differ significantly in how they balance neural intuition with symbolic rigor:

* **AlphaGeometry (Nature 2024):** This system combines a neural language model with a symbolic deduction engine. The LLM is responsible for "constructive moves" (adding auxiliary lines or points), while the symbolic engine deduces new statements from these constructions. It solves Olympiad-level problems without human demonstrations.

* **The FGeo Series:** This suite of models explores various architectures for geometric reasoning:
    * **FGeo-TP:** Enhances a symbolic solver with a Language Model to guide the proof process.
    * **FGeo-HyperGNet:** Integrates a formal symbolic system with a Hypergraph Neural Network to better represent geometric relationships.
    * **FGeo-DRL:** Utilizes Deep Reinforcement Learning to perform deductive reasoning.
    * **FGeo-SSS:** A search-based symbolic solver focused on human-like reasoning.

* **Symbolic Baselines (Wu's Method):** Recent work by Sinha et al. (2024) challenges the necessity of heavy neural components for certain problems. They demonstrate that **Wu's Method** (a classic algebraic geometry algorithm), when properly implemented, can rival silver medalists and even outperform AlphaGeometry on specific IMO geometry benchmarks.




## Autoformalization and the Data Bottleneck
A critical bottleneck in AI4Math is the scarcity of formalized data—most mathematical knowledge exists as informal LaTeX or textbook prose, which computers cannot verify. **Autoformalization** is the task of automatically translating natural language math into formal code (e.g., Lean).

* **Wu et al. (2022):** Demonstrated that LLMs could be few-shot prompted to perform this translation.
* **ProofNet (2023) and Lean Workbook (2024):** Provided benchmarks to evaluate how well models can formalize undergraduate-level mathematics.
* **TheoremLlama (2024):** Represents the recent trend of fine-tuning "expert" models specifically for this translation task to populate formal libraries.



## Synthetic Discovery and Intuition
Beyond verifying known truths, AI is increasingly used to discover new mathematics.

* **Synthetic Theorem Generation:** Models are trained not just to prove, but to propose interesting conjectures. **MetaGen (2020)** and **MUSTARD (2024)** focused on learning to generate theorems, creating a curriculum of synthetic data to train stronger provers.
* **AI for Intuition:** The **Nature 2021 paper by Davies et al.** showcased how AI could guide human intuition, helping mathematicians discover patterns in knot theory and representation theory that were previously unnoticed.
* **FunSearch (Nature 2024):** Used LLMs to search for functions in code space, discovering new constructions in combinatorics (e.g., the cap set problem) that surpassed best-known human results.


## Benchmarks and Evaluation
To measure progress, the field has coalesced around several key benchmarks:

* **MiniF2F (2022):** A cross-system benchmark of Olympiad-level problems, becoming the standard for comparing neural provers.
* **FIMO (2023):** A challenge dataset specifically for formal automated theorem proving, pushing the difficulty closer to International Mathematical Olympiad standards.



## Summary
The literature is transiting from "AI as a search heuristic" to "AI as a generative partner." The integration of Large Language Models with formal verification (as seen in *LeanDojo* and *DeepSeek-Prover*) and the specific success in geometry (*AlphaGeometry*) suggests that the future of AI4Math lies in *neuro-symbolic systems*: models that possess the linguistic fluency to conjecture and autoformalize, paired with the rigorous logic of proof assistants to verify and guide their reasoning.\\
\\

----

## References

1.  **Advancing mathematics by guiding human intuition with AI.** *Nature 2021* [[pdf](https://www.nature.com/articles/s41586-021-04086-x)]
    *Alex Davies, Petar Veličković, Lars Buesing, Sam Blackwell, Daniel Zheng, Nenad Tomašev, Richard Tanburn, Peter Battaglia, Charles Blundell, András Juhász, Marc Lackenby, Geordie Williamson, Demis Hassabis & Pushmeet Kohli*

2.  **Autoformalization with Large Language Models.** *NeurIPS 2022* [[pdf](https://openreview.net/pdf?id=IUikebJ1Bf0)]
    *Yuhuai Wu, Albert Qiaochu Jiang, Wenda Li, Markus Norman Rabe, Charles E Staats, Mateja Jamnik, Christian Szegedy*

3.  **Baldur: Whole-Proof Generation and Repair with Large Language Models** *arxiv preprint 2023* [[pdf](https://arxiv.org/pdf/2303.04910.pdf)]
    *Emily First, Markus N. Rabe, Talia Ringer, Yuriy Brun*

4.  **DeepMath - Deep Sequence Models for Premise Selection.** *NeurIPS 2016* [[pdf](https://proceedings.neurips.cc/paper/2016/file/f197002b9a0853eca5e046d9ca4663d5-Paper.pdf)]
    *Alex A. Alemi, Francois Chollet, Niklas Een, Geoffrey Irving, Christian Szegedy, Josef Urban*

5.  **DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data.** *arXiv preprint 2024* [[pdf](https://arxiv.org/pdf/2405.14333)]
    *Huajian Xin, Daya Guo, Zhihong Shao, Zhizhou Ren, Qihao Zhu, Bo Liu, Chong Ruan, Wenda Li, Xiaodan Liang*

6.  **FGeo-DRL: Deductive Reasoning for Geometric Problems through Deep Reinforcement Learning.** *Symmetry 2024* [[pdf](https://arxiv.org/pdf/2402.09051)]
    *Jia Zou, Xiaokai Zhang, Yiming He, Na Zhu, Tuo Leng*

7.  **FGeo-HyperGNet: Geometry Problem Solving Integrating Formal Symbolic System and Hypergraph Neural Network.** *arXiv preprint 2024* [[pdf](https://arxiv.org/pdf/2402.11461)][[code](https://github.com/BitSecret/HyperGNet)]
    *Xiaokai Zhang, Na Zhu, Yiming He, Jia Zou, Cheng Qin, Yang Li, Zhenbing Zeng, Tuo Leng*

8.  **FGeo-SSS: A Search-Based Symbolic Solver for Human-like Automated Geometric Reasoning.** *Symmetry 2024* [[pdf](https://www.mdpi.com/2073-8994/16/4/404)]
    *Xiaokai Zhang, Na Zhu, Yiming He, Jia Zou, Cheng Qin, Yang Li, Tuo Leng*

9.  **FGeo-TP: A Language Model-Enhanced Solver for Geometry Problems.** *Symmetry 2024* [[pdf](https://arxiv.org/pdf/2402.09047)]
    *Yiming He, Jia Zou, Xiaokai Zhang, Na Zhu, Tuo Leng*

10. **FIMO: A Challenge Formal Dataset for Automated Theorem Proving.** *arXiv preprint 2023* [[pdf](https://arxiv.org/pdf/2309.04295.pdf)]
    *Chengwu Liu, Jianhao Shen, Huajian Xin, Zhengying Liu, Ye Yuan, Haiming Wang, Wei Ju, Chuanyang Zheng, Yichun Yin, Lin Li, Ming Zhang, Qun Liu*

11. **Generative Language Modeling for Automated Theorem Proving.** *arXiv preprint 2020* [[pdf](https://arxiv.org/pdf/2009.03393.pdf)]
    *Stanislas Polu, Ilya Sutskever*

12. **HOList: An Environment for Machine Learning of Higher Order Logic Theorem Proving.** *ICML 2019* [[pdf](http://proceedings.mlr.press/v97/bansal19a/bansal19a.pdf)] [[dataset](https://sites.google.com/view/holist/home)]
    *Kshitij Bansal, Sarah Loos, Markus Rabe, Christian Szegedy, Stewart Wilcox*

13. **Holophrasm: a neural Automated Theorem Prover for higher-order logic.** *arXiv preprint 2016* [[pdf](https://arxiv.org/pdf/1608.02644.pdf)]
    *Daniel Whalen*

14. **Lean Workbook: A large-scale Lean problem set formalized from natural language math problems.** *arXiv preprint 2024* [[pdf](https://arxiv.org/pdf/2406.03847)] [[dataset](https://huggingface.co/datasets/internlm/Lean-Workbook)] [[code](https://github.com/InternLM/InternLM-Math)]
    *Huaiyuan Ying, Zijian Wu, Yihan Geng, Jiayu Wang, Dahua Lin, Kai Chen*

15. **LeanDojo: Theorem Proving with Retrieval-Augmented Language Models.** *NeurIPS 2023 Datasets and Benchmarks Track* [[pdf](https://arxiv.org/pdf/2306.15626.pdf)] [[code](https://github.com/lean-dojo)]
    *Kaiyu Yang, Aidan M. Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan Prenger, Anima Anandkumar*

16. **Learning to Prove Theorems by Learning to Generate Theorems.** (MetaGen) *NeurIPS 2020* [[pdf](https://proceedings.neurips.cc/paper/2020/file/d2a27e83d429f0dcae6b937cf440aeb1-Paper.pdf)] [[code](https://github.com/princeton-vl/MetaGen)]
    *Mingzhe Wang, Jia Deng*

17. **Mathematical discoveries from program search with large language models.** (FunSearch) *Nature 2024* [[pdf](https://www.nature.com/articles/s41586-023-06924-6)][[code](https://github.com/google-deepmind/funsearch)]
    *Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, Alhussein Fawzi*

18. **MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics.** *ICLR 2022* [[pdf](https://openreview.net/pdf?id=9ZPegFuFTFv)] [[dataset](https://github.com/openai/miniF2F)]
    *Kunhao Zheng, Jesse Michael Han, Stanislas Polu*

19. **MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data.** *ICLR 2024* [[pdf](https://openreview.net/pdf?id=8xliOUg9EW)][[code](https://github.com/Eleanor-H/MUSTARD)]
    *Yinya Huang, Xiaohan Lin, Zhengying Liu, Qingxing Cao, Huajian Xin, Haiming Wang, Zhenguo Li, Linqi Song, Xiaodan Liang*

20. **ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics.** *arXiv preprint 2023* [[pdf](https://arxiv.org/pdf/2302.12433.pdf)] [[code](https://github.com/zhangir-azerbayev/proofnet)]
    *Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward W. Ayers, Dragomir Radev, Jeremy Avigad*

21. **Solving olympiad geometry without human demonstrations.** (AlphaGeometry) *Nature 2024* [[pdf](https://www.nature.com/articles/s41586-023-06747-5)][[code](https://github.com/google-deepmind/alphageometry)]
    *Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, Thang Luong*

22. **TacticZero: Learning to Prove Theorems from Scratch with Deep Reinforcement Learning.** *NeurIPS 2021* [[pdf](https://openreview.net/pdf?id=edmYVRkYZv)]
    *Minchao Wu, Michael Norrish, Christian Walder, Amir Dezfouli*

23. **TheoremLlama: Transforming General-Purpose LLMs into Lean4 Experts.** *EMNLP 2024* [[pdf](https://arxiv.org/pdf/2407.03203)] [[code](https://github.com/RickySkywalker/TheoremLlama)]
    *Ruida Wang, Jipeng Zhang, Yizhen Jia, Rui Pan, Shizhe Diao, Renjie Pi, Tong Zhang*

24. **Thor: Wielding Hammers to Integrate Language Models and Automated Theorem Provers.** *NeurIPS 2022* [[pdf](https://openreview.net/pdf?id=fUeOyt-2EOp)]
    *Albert Q. Jiang, Wenda Li, Szymon Tworkowski, Konrad Czechowski, Tomasz Odrzygóźdź, Piotr Miłoś, Yuhuai Wu, Mateja Jamnik*

25. **Wu's Method can Boost Symbolic AI to Rival Silver Medalists and AlphaGeometry to Outperform Gold Medalists at IMO Geometry.** *arXiv preprint 2024* [[pdf](https://arxiv.org/pdf/2404.06405)][[code](https://huggingface.co/datasets/bethgelab/simplegeometry)]
    *Shiven Sinha, Ameya Prabhu, Ponnurangam Kumaraguru, Siddharth Bhat, Matthias Bethge*


