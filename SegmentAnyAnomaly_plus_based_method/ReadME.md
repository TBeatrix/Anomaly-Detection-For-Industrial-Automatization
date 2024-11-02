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
