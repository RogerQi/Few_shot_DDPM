# Few_shot_DDPM

This implementation is minimalistic and would likely require no extra dependencies if you have PyTorch working.

## Runnning the code

```
python train_diffusion.py
python train_classifier.py

python eval_diffusion.py  # for unconditional synthesis
python classifier_sample.py  # for conditional synthesis
```

It is default to working on FashionMNIST. If you would like to try out other datasets, modify `params.py`.

