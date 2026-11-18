# Architecture notes

## The land-masking bug was the main flaw

The DDPM's main flaw was a land-masking mismatch between training and sampling.
Training zeroed the land pixels, but the sampling chain did not, so the network
saw land values during sampling that it had never seen in training.

This is now fixed: the sampling chain zeroes land in the same way as training.
See `predict_x0` in [models/ddpm.py](../models/ddpm.py).

**Consequence:** the issues we earlier treated as the root causes were much less
important than this land error. They include reverse-chain drift, exposure bias,
noise-schedule effects, and melt/cool season bias. Results and diagnoses from
before the fix are confounded by the bug. Re-check them on the fixed code before
building on them.
