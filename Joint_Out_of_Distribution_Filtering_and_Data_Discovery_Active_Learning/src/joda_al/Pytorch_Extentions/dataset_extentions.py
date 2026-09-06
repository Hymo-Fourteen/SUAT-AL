from typing import Sequence, Optional, Callable, TypeVar, Generic, Any, Tuple

from torch.utils.data import Subset, Dataset
from torch.utils.data.dataset import T_co
import inspect

class TransformSubset(Subset):
    """
    A subset of a dataset that applies optional transforms to each sample.

    Args:
        dataset (Dataset): The dataset to subset.
        indices (Sequence[int]): Indices to include in the subset.
        transform (Optional[Callable]): Optional transform to apply to the input (img).
        transforms (Optional[Callable]): Optional transform to apply to both input and target.

    Returns:
        Tuple[Any, Any]: Transformed input and target.
    """
    def __init__(
        self,
        dataset: Dataset[T_co],
        indices: Sequence[int],
        transform: Optional[Callable[[Any], Any]] = None,
        transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None
    ) -> None:
        super().__init__(dataset, indices)
        self.transform = transform
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def to_TransformDataset(self) -> 'TransformDataset':
        """
        Converts this TransformSubset into a TransformDataset.

        Returns:
            TransformDataset: A new TransformDataset wrapping the same dataset and transforms.
        """
        return TransformDataset(self.dataset, self.transform, self.transforms)


class TransformSubsetLightning(Subset):
    """
    A subset for Lightning that applies an optional transform to each sample.

    Args:
        dataset (Dataset): The dataset to subset.
        indices (Sequence[int]): Indices to include in the subset.
        transform (Optional[Callable]): Optional transform to apply to the sample.
        transforms (Optional[Callable]): Optional transform to apply to the sample.

    Returns:
        Any: Transformed sample.
    """
    def __init__(
        self,
        dataset: Dataset[T_co],
        indices: Sequence[int],
        transform: Optional[Callable[[Any], Any]] = None,
        transforms: Optional[Callable[[Any], Any]] = None
    ) -> None:
        super().__init__(dataset, indices)
        self.transform = transform
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Any:
        result = super().__getitem__(idx)
        if self.transforms:
            try:
                result = self.transforms(result)
            except Exception:
                result = self.transforms(result)
        return result


T = TypeVar("T", bound=Dataset, covariant=True)

class TransformDataset(Generic[T]):
    """
    A dataset wrapper that applies optional transforms to each sample.

    Args:
        dataset (Dataset): The dataset to wrap.
        transform (Optional[Callable]): Optional transform to apply to the input (img).
        transforms (Optional[Callable]): Optional transform to apply to both input and target.

    Returns:
        Tuple[Any, Any]: Transformed input and target.
    """
    def __init__(
        self,
        dataset: T,
        transform: Optional[Callable[[Any], Any]] = None,
        transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None
    ) -> None:
        # Copy all methods and attributes from the original dataset, unless already present
        for method in inspect.getmembers(dataset, predicate=inspect.ismethod):
            if method[0] not in self.__class__.__dict__:
                setattr(self, method[0], method[1])
        for method in inspect.getmembers(dataset, predicate=inspect.isfunction):
            if method[0] not in self.__class__.__dict__:
                setattr(self, method[0], method[1])
        for name, param in dataset.__dict__.items():
            if name not in self.__class__.__dict__:
                setattr(self, name, param)
        self.dataset = dataset
        self.transform = transform
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.dataset.__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self) -> int:
        return len(self.dataset)

    def to_TransformSubset(
        self,
        indices: Sequence[int]
    ) -> TransformSubset:
        """
        Converts this TransformDataset into a TransformSubset.

        Args:
            indices (Sequence[int]): Indices to include in the subset.

        Returns:
            TransformSubset: A new TransformSubset with the same transforms.
        """
        return TransformSubset(self, indices, self.transform, self.transforms)
