# LLE-INC
Repository for AAAI 2024 paper ["Manifold-based Verbalizer Space Re-embedding for Tuning-free Prompt-based Classification"](https://arxiv.org/abs/2309.04174)

## Requirements

Please refer to the file  ```requirements.yaml``` for the required packages. 

## A quick start

### Prepare the Data
English Few-shot datasets can be downloaded [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). 
Chinese datasets can be downloaded with applications on the websites of the corresponding benchmarks. 

### Get the Logits
Get the representation of the [MASK] token. 
Run the  ```infer_get_logits.py``` with the proper file path of datasets and models. 
An example is given with the model of LLaMA and dataset of SST-2.

### Contrastive Learning (Optional)

As the paper stated, contrastive learning works as a complementary module for the language models and is optional.
The code is adopted from the [prototypical verbalizer](https://github.com/thunlp/OpenPrompt/blob/main/openprompt/prompts/prototypical_verbalizer.py).

### Verbalizer Space Re-embedding
1. Make sure the version of ```sklearn``` is consistent with that in the yaml file.
2. Enter the path of installed sklearn package

	> something like '/.../lib/python3.x/site-packages/sklearn/manifold'

3. Replace the ```__init__.py``` with the file in the ```LLE-INC``` folder.
4. Copy the ```_locally_linear_mod.py``` into the ```manifold``` folder.
5. Run the re_embedding.py with the correct data path (pickle file for the instance representation) with the same 
format as that in the ```logits_example.pickle```. Note that the performance can be fluctuated with the hyper-parameters.

## Citation
If you find our work useful, please cite the following arxiv paper for now (since the proceedings of AAAI 2024 have not been released).:
```bibtex
@article{wang2023manifold,
  title={Manifold-based Verbalizer Space Re-embedding for Tuning-free Prompt-based Classification},
  author={Wang, Haochun and Zhao, Sendong and Liu, Chi and Xi, Nuwa and Cai, Muzhen and Qin, Bing and Liu, Ting},
  journal={arXiv preprint arXiv:2309.04174},
  year={2023}
}
```

## Contact us
If you have any question about our paper or code, feel free to contact me with ```hcwang@ir.hit.edu.cn```.