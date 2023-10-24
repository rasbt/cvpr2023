# Saving and Loading Models in Fabric



There are several ways to save and load modules in Fabric. 



The simplest way to load an existing, saved PyTorch model is to use the `load_raw`:

```python
fabric = Fabric()
model = MyModel()

# model.pt is a regular PyTorch state_dict via torch.save
fabric.load_raw("path/to/model.pt", model)
```

However, Fabric has a `load` function that can save additional information to a checkpoint file, for example, the optimizer state, which makes it incredibly useful.

Below, I will illustrate 3 common scenarios for saving and loading models with Fabric:


&nbsp;
## 1) Train on 1 GPU, load on 1 GPU

Suppose you saved a model after training as follows:

```python
# training.py file

fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=1)
fabric.launch()
# ...
state = {
    "model": model,
    "optimizer": optimizer,
    "anything-else-you-want-to-save": 123
}
fabric.save("checkpoint.ckpt", state)
```

Then you can load it in a separate file as follows:

```python
# loading.py file
fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=1)
fabric.launch()
model, optimizer = fabric.setup(model, optimizer)

state = {
    "model": model,
    "optimizer": optimizer,
    "anything-else-you-want-to-save": None
}

fabric.load("checkpoint.ckpt", state)
```


&nbsp;
## 2) Train on 4 GPUs, load on 4 GPUs

Suppose you trained a model in a distributed fashion and saved it as follows:

```python
# training.py file

fabric = Fabric(
    accelerator="cuda", precision="bf16-mixed",
    devices=4, strategy="fsdp"
)
fabric.launch()
# ...
state = {
    "model": model,
    "optimizer": optimizer,
    "anything-else-you-want-to-save": 123
}
fabric.save("checkpoint.ckpt", state)
```

This will automatically shard the checkpoint file as well; here it will create a checkpoint folder with 4 smaller checkpoint chunks since we used 4 devices.

Then you can load it it as follows:

```python
# loading.py file
fabric = Fabric(
    accelerator="cuda", precision="bf16-mixed",
    devices=4, strategy="fsdp"
)
fabric.launch()
model, optimizer = fabric.setup(model, optimizer)

state = {
    "model": model,
    "optimizer": optimizer,
    "anything-else-you-want-to-save": None
}

fabric.load("checkpoint.ckpt", state)
```

&nbsp;
## 3) Train on 4 GPUs, load on 1 GPUs

The maybe most common scenario is to train a model on multiple GPUs, and then use the model on a single GPU for inference later. As mentioned above, Fabric saves distributed checkpoint chunks by default. You can change this behavior and create a single checkpoint file via the `state_dict_type="full"` shown below:



```python
# training.py file
strategy = FSDPStrategy(state_dict_type="full")
fabric = Fabric(
    accelerator="cuda", precision="bf16-mixed",
    devices=4, strategy=strategy
)
fabric.launch()
# ...
state = {
    "model": model,
    "optimizer": optimizer,
    "anything-else-you-want-to-save": 123
}
fabric.save("checkpoint.ckpt", state)
```

Then you can load the checkpoint as follows:

```python
# loading.py file
fabric = Fabric(
    accelerator="cuda", precision="bf16-mixed", devices=1)
fabric.launch()
model, optimizer = fabric.setup(model, optimizer)

state = {
    "model": model,
    "optimizer": optimizer,
    "anything-else-you-want-to-save": None
}

fabric.load("checkpoint.ckpt", state)
```

If you want to give this a try, this third way is implemented in the [08-1-train.py](08-1-train.py) and [08-2-load.py](08-2-load.py) scripts.