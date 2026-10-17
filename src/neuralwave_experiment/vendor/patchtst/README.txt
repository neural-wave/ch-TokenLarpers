This package intentionally includes LICENSE, NOTICE, and README.txt as package data so provenance remains accessible in built distributions.
PatchTST upstream source attribution
====================================

Source: https://github.com/yuqinie98/PatchTST
Pinned commit: 204c21efe0b39603ad6e2ca640ef5896646ab1a9
License: Apache License 2.0 (see LICENSE)

This distribution contains the self-supervised PatchTST source subset:
models/patchTST.py, models/layers/attention.py, models/layers/basics.py,
and models/layers/pos_encoding.py. Relative imports were repaired for this
package. The encoder and attention boundaries were modified to propagate key
padding masks; the adapter adds NeuralWave calendar/observation projections.
Those modifications are not part of the untouched upstream implementation.
