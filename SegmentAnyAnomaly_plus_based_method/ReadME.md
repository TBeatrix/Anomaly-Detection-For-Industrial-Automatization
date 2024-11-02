SegmentAnyAnomaly model:

@article{cao_segment_2023,
	title = {Segment Any Anomaly without Training via Hybrid Prompt Regularization},
	url = {http://arxiv.org/abs/2305.10724},
	number = {{arXiv}:2305.10724},
	publisher = {{arXiv}},
	author = {Cao, Yunkang and Xu, Xiaohao and Sun, Chen and Cheng, Yuqi and Du, Zongwei and Gao, Liang and Shen, Weiming},
	urldate = {2023-05-19},
	date = {2023-05-18},
	langid = {english},
	eprinttype = {arxiv},
	eprint = {2305.10724 [cs]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Artificial Intelligence},
}

@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@inproceedings{ShilongLiu2023GroundingDM,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and Jie Yang and Chunyuan Li and Jianwei Yang and Hang Su and Jun Zhu and Lei Zhang},
  year={2023}
}

# Useage of SAA+ for the industrial products

1. Download SAA+ from the original implementation
2. Follow the insallation proccess, detailed in the original implementation
or
use the provided Dockerfile and the requirements_SAA.txt for a Docker environment

3.  Copy the dataset in the root directory

4. Change the following files from the ones provided in this repository:
- eval_SAA.py (SegmentAnyAnomaly/)   
- hybrid_prompts (SegmentAnyAnomaly/SAA/)

5. Add the following files from this repository:
- product_parameters.py         into SegmentAnyAnomaly/SAA/prompts
- products.py                   into SegmentAnyAnomaly/datasets
- run_industrial_products.py    into SegmentAnyAnomaly

6. New the pediction can be run with:  python3 run_industrial_products.py 
and the results will be in the configured directory
