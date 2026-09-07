```markdown
# Selection Methods

This folder contains various selection methods used in the active learning framework. These methods are responsible for selecting the most informative samples from the unlabeled dataset to be labeled and added to the training set.

## Contents

- `query_method_def.py`: Contains the definition of the `QueryMethod` abstract class and the `QueryFactory` class.
- `diversity_based.py`: Contains implementations of diversity-based selection methods.

## Usage

To use the selection methods in this folder, you need to import the desired method into your active learning pipeline. Below is an example of how to import and use a selection method.

### Example

```python
from Selection_Methods import QueryFactory

# Assuming you have a DataLoader for your unlabeled data
unlabeled_loader = ...

# Initialize your selection method using the factory
selection_method = QueryFactory.get_query("GradCoreSet")

# Use the selection method to select samples
selected_samples = selection_method.query(models, handler, dataset_handler, config, query_size, device)
```

### QueryFactory

The `QueryFactory` class is used as an entry point to register and retrieve query methods. It maintains a registry of available query methods and provides a method to retrieve them by name.

### QueryMethod Abstract Class

The `QueryMethod` abstract class defines the interface for all selection methods. It includes an abstract `query` method that must be implemented by all subclasses. The `query` method has the following parameters:

- `models` (Dict): A dictionary of models to be used in the query.
- `handler` (Union[TaskHandler, ModelHandler]): An instance of a task handler or model handler.
- `dataset_handler` (DataSetHandler): An instance of a dataset handler.
- `config` (Dict): Configuration settings for the training process.
- `query_size` (int): The desired size of the query.
- `device`: The device on which the query will be executed (e.g., 'cpu', 'cuda').
- `labeled_idx_set` (Optional[int]): Indices of the labeled dataset.
- `unlabeled_idx_set` (Optional[int]): Indices of the unlabeled dataset.
- `*arg`: Variable-length argument list.
- `**kwargs`: Arbitrary keyword arguments.

The `query` method returns a tuple containing two lists:
- The first list contains values justifying the selection.
- The second list contains the unlabeled indices for selection sorted by importance in descending order.

### Adding New Selection Methods

To add a new selection method, create a new Python file in this folder and define your selection method class. Ensure that your class implements the necessary interface for selection methods.

### Example of a New Selection Method

```python
# File: Selection_Methods/my_selection_method.py

class MySelectionMethod(QueryMethod):
    keywords = ["MyMethod"]

    def query(self, models, handler, dataset_handler, config, query_size, device, labeled_idx_set=None, unlabeled_idx_set=None, *arg, **kwargs):
        # Implement your selection logic
        selected_samples = ...
        return selected_samples
```

After creating your new selection method, you can register it with the `QueryFactory` and use it in your active learning pipeline as shown in the example above.

