# WiserUI-Bench

<h2 align="center">
Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding
</h2>

<p align="center">
Authors: <strong>Jaehyun Jeon</strong>, Min Soo Kim, Jang Han Yoon, Sumin Shim, Yejin Choi, Hanbin Kim, Youngjae Yu
</p align="center">

<p align="center">
<a href='https://arxiv.org/abs/2505.05026'><img src='https://img.shields.io/badge/Arxiv-2505.05026-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
<a href='https://huggingface.co/datasets/jeochris/WiserUI-Bench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
</p>

![main_figure](figure/main.png)

We introduce **WiserUI-Bench**, a benchmark for evaluating MLLMs’ understanding of user behavior-oriented UI/UX design using real-world A/B-tested interfaces and expert-curated key rationales. Results show current models struggle with nuanced reasoning about UI/UX design and its behavioral impact. For further details, please check out our [paper](https://arxiv.org/abs/2505.05026).

## Inference
We provide an inference framework for (1) UI/UX design selection and (2) UI/UX design rationale alignment task on [WiserUI-Bench](https://huggingface.co/datasets/jeochris/WiserUI-Bench).

```
pip install -r requirements.txt
cd inference
bash execute.sh
```

- For `execute.sh`,
  - Input your model / method / task number / gpu count based on your needs.
  - Input your OpenAI/Claude API key if needed.
- All the open-source models we used are supported by `vllm`.

## Code Structure

```
inference/
├── prompts_task1/  # Prompts for Task 1 (selection)
├── prompts_task2/  # Prompts for Task 2 (explanation)
├── task.py         # entry-point on WiserUI-Bench
├── methods.py      # handling prompting methods
└── VLM.py          # model inference wrapper
```

You can also use your custom prompts, placing in prompts folder.

## Supported Models

We support inference with the following models currently:
- Proprietary : o1, GPT-4o, Claude 3.5 Sonnet
- Open-source : Qwen-2.5-VL (7B, 32B), InternVL-2.5 (8B, 38B), LLaVA-NeXT 7B, LLaVA-OneVision 7B

You can also use your own models by modifying the provided code.

## TODO  
- [] Release evaluation code
- [] Release mechanism supporting custom A/B-tested UI datasets 

## Citation
If you find our project useful, please cite:
```
@misc{jeon2025mllmscaptureinterfacesguide,
      title={Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding}, 
      author={Jaehyun Jeon and Min Soo Kim and Jang Han Yoon and Sumin Shim and Yejin Choi and Hanbin Kim and Youngjae Yu},
      year={2025},
      eprint={2505.05026},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.05026}, 
}
```