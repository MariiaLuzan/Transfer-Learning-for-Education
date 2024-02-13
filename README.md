# Transfer-Learning-for-Education

**Evaluation of transfer learning methods for predictive tasks in education**


## Motivation
Student dropout is a big issue in higher education. To tackle it, one approach is to identify struggling students early and support them. This involves using machine learning to find students at risk of dropping out. However, some universities lack resources, like data or data scientists, to develop these models. This gap deepens disparities between well-resourced and less resourceful institutions. 

Transfer learning may present a solution for universities with limited resources, enabling them to leverage models or data from more resourceful universities, bridging the resource gap, as demonstrated in the study by *Joshua Gardner, Renzhe Yu, Quan Nguyen, Christopher Brooks, and Rene Kizilcec titled "Cross-institutional transfer learning for educational models: Implications for model performance, fairness, and equity." (FAccT ’23).* [Link to the paper](https://doi.org/10.1145/3593013.3594107). 

The aim of this project is to assess how well transfer learning methods perform for this particular task.


## Dataset
The dataset consists of anonymized student information from a major U.S. public university. It covers various static details about students, like demographics, socio-economic status, and high school academic performance. Additionally, it includes dynamic term-specific university data, such as academic career, grades, and workload details.


## Experiment
Due to limited access to data from only one university, a transfer learning scenario was simulated. Two colleges within the university were selected to represent the source and target institutions.


## Transfer learning methods for evaluation
1.	**Instance weighted strategy based on kernel mean matching** (*Ref: Jiayuan Huang, Arthur Gretton, Karsten Borgwardt, Bernhard Schölkopf, and Alex Smola. Correcting sample selection bias by unlabeled data. In Advances in Neural Information Processing Systems, volume 19. MIT Press, 2006.* [Link to the paper](https://proceedings.neurips.cc/paper_files/paper/2006/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf) ).
2.	**TrAdaBoost** (*Ref: Wenyuan Dai, Qiang Yang, Gui-Rong Xue, and Yong Yu "Boosting for transfer learning" (ICML '07).* [Link to the paper](https://doi.org/10.1145/1273496.1273521) ), but with several modifications.


## Evaluation of the transfer learning methods
The performance of the assessed transfer learning methods is compared with direct transfer, where a model trained on the source dataset is directly applied to the target dataset without modifications. The performance is evaluated using classification accuracy and fairness metrics.

## Code execution guide
1. Run the '1. Sample.ipynb' notebook first, as it prepares a sample used in all subsequent notebooks.
2. Notebooks can then be run in any order following the initial execution of '1. Sample.ipynb'.
