# TRIM-KV: Token Retention for Memory-Bounded Key-Value Eviction

<a href="https://arxiv.org/pdf/2512.03324"><img src="https://img.shields.io/badge/arxiv-2512.03324-red?style=for-the-badge"></a>

### What is TRIM-KV?

> An efficient and learnable key–value eviction strategy designed to improve the efficiency of large language models (LLMs) in long-horizon inference.

Imagine if our brain worked like a transformer:

<div align="center">
    <img width="1000" alt="teaser" src="assets/fun.gif"/>
</div>

Our brain will explode 🧠💥 and so would your GPU. TRIM-KV lets your model forget the parts that don’t matter much, so it doesn’t melt its VRAM. Don’t let the brain explode.

The core idea behind TRIM-KV is to learn the intrinsic importance of each key–value pair at creation time, what we call *token retention*, and then decay this importance exponentially over time, mimicking standard inference with eviction.

The retention score is query-agnostic and captures the long-term utility of tokens. This is different from attention scores, which are query-dependent: they capture the short-term utility for predicting the next token and are recomputed at every step, making them local, myopic, and highly dependent on the transient decoding state.

In case you find this useful

```tex
@article{bui2025cache,
  title={Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs},
  author={Bui, Ngoc and Sharma, Shubham and Lamba, Simran and Mishra, Saumitra and Ying, Rex},
  journal={arXiv preprint arXiv:2512.03324},
  year={2025}
}
```


### Why TRIM-KV?

It's fast

<div align="center">
    <img width="1000" alt="teaser" src="assets/speed.png"/>
</div>

It's smart

<div align="center">
    <img width="1000" alt="teaser" src="assets/performance.png"/>
</div>


And it's interpretable

<div align="center">
    <img width="1000" alt="teaser" src="assets/eviction.png"/>
</div>

<div align="center">
    <img width="1000" alt="teaser" src="assets/vis.png"/>
</div>

One of the most interesting observations we found is that, in some layers, TRIM-KV tends to retain only period tokens. This suggests that, in these heads, period tokens may implicitly act as gist tokens that summarize information from the preceding sentence. This contrasts with recent approaches that advocate retaining entire chunks of tokens. Our results indicate that it can be more budget-efficient to keep carefully chosen individual tokens, since they already carry rich contextual information.

Please refer to Section 5.1.2 in our paper for detailed discussions. Check [`examples/visualize.ipynb`](examples/visualize.ipynb) for the visualization script.

---

### Some Notes

- I’m waiting for approval before releasing the code.

- We recently refactored the code to support transformers v4.57.0. This release is to freeze the codebase at a version close to what was used in the paper to ensure all reported results are reproducible. If you encounter any issues, please open a GitHub issue.

- Further refactoring and updates are planned in the near future.

---

## Acknowledgements

A large portion of this repository is adapted from or built upon the following projects:

* [https://github.com/microsoft/SeerAttention](https://github.com/microsoft/SeerAttention)
* [https://github.com/Zefan-Cai/R-KV](https://github.com/Zefan-Cai/R-KV)
* [https://github.com/microsoft/MInference](https://github.com/microsoft/MInference)
* [https://github.com/huangyuxiang03/Locret](https://github.com/huangyuxiang03/Locret)
* [https://github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)
* [https://github.com/princeton-pli/LongProc](https://github.com/princeton-pli/LongProc)
* [https://github.com/THUDM/LongBench](https://github.com/THUDM/LongBench)
