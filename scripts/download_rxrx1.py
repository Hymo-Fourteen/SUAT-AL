from wilds import get_dataset

if __name__ == "__main__":
    ds = get_dataset("rxrx1", root_dir="./data/wilds", download=True)
    print("data dir:", ds.data_dir)
    print("splits:", {s: len(ds.get_subset(s)) for s in ["train", "val", "test", "id_test"]})
    print("classes:", ds.n_classes)