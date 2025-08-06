# WiserUI-Bench

<h2 align="center">
Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding
</h2>

**Authors:** **Jaehyun Jeon**, Min Soo Kim, Jang Han Yoon, Sumin Shim, Yejin Choi, Hanbin Kim, Youngjae Yu

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