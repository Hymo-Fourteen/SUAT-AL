
# How to Run Joint Out-of-Distribution Filtering and Data Discovery Active Learning

Joint Out-of-Distribution Filtering and Data Discovery Active Learning is an active learning method designed to handle open-set discovery scenarios. This approach is particularly useful when dealing with datasets where new, unseen classes may appear, requiring the model to adapt and learn effectively.

## Exemplary Command

To execute Joint Out-of-Distribution Filtering and Data Discovery Active Learning with specific configurations, use the following command:

```bash
python3 src/joda_al/start_active_learning.py --method-type Joda --scenario osal-extending --loss-func OutlierExposure --triplet-mode triplet --use-pca False --seperator metric --sep-metric energy --coverage-method DSC --surprise-strategy surprise --coverage-al incremental --gain-mode energy --ind-classes '[0,1,2]' --near-classes '[3,4,5,6,7,8,9]' --far-dataset random --sigmoids '{"AdaptiveAvgPool2d-1":100,"Sequential-3":1000,"Sequential-2":0.001,"Sequential-1":0.001}'
```

### Relevant Parameters

- **scenario**: `osal-extending` indicates open-set discovery active learning, where out-of-distribution (OOD) classes are split into near and far OOD. Near OOD classes can be added to the in-distribution (IND) data if enough samples are collected.
- **loss-func**: `OutlierExposure` regularizes OOD data to output a uniform distribution. It is used with `triplet-mode` "triplet" in the `osal-extending` scenario.
- **triplet-mode**: When set to `triplet`, it filters out far OOD data and uses a triplet loss consisting of IND, near OOD, and far OOD samples.

### Dataset Settings

- **ind-classes**: Specifies the list of IND classes.
- **near-classes**: List of near OOD classes. If empty, it uses the remaining classes not listed as IND.
- **far-classes**: If not specified, a `far-dataset` must be provided.
- **far-dataset**: Specifies the dataset used as far OOD data, such as `random`, `mnist`, `places365F`, or any dataset from OpenOOD.


Predefined configurations can be used for specific datasets:

```bash
python3 src/joda_al/start_active_learning.py ... --config-path Stream_Based_AL/Config/json/TinyImageNet60.json ...
python3 src/joda_al/start_active_learning.py ... --config-path Stream_Based_AL/Config/json/Cifar100-60.json ...
```

## Run Experiments

To run experiments with specific settings, use the following command:

```bash
python3 src/joda_al/start_active_learning.py --method-type <MethodType> --scenario osdal --dataset cifar100-ta --ind-classes '[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]' --near-classes '[]' --query-size 2500 --is-percent False --far-dataset places365F --num-epochs 200 --batch-size 128 --opensetmode InD --base Pool
```

## Settings Methods

### Joint Out-of-Distribution Filtering and Data Discovery Active Learning

```bash
--method-type Joda
--loss-func OutlierExposure
--triplet-mode triplet
--opensetmode All
```

### AOL

```bash
--method-type aol
--loss-func OpenCrossEntropy
--triplet-mode triplet
--opensetmode All
```

### MQNet & CCAL

```bash
--method-type mqnet::ccal
--triplet-mode off
--opensetmode InD
```

### Classic AL

```bash
--method-type Random::cEnt::llossOrg
--opensetmode InD
```

### LfOSA

```bash
--method-type LfOSA
--scenario osal-extending
--dataset cifar100-ta
--triplet-mode triplet
--loss-func OODCrossEntropy
--opensetmode All
```

### Pal

```bash
--method-type Pal
--triplet-mode off
--model OpenResNet18
```

These settings and commands allow for a flexible approach to active learning, adapting to various datasets and scenarios, ensuring robust model training and evaluation.
