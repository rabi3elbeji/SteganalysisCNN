# Model architecture
![alt text](images/model.png "Proposed model")

This work inspired by a recent work of [Mo Chen et al](http://www.ws.binghamton.edu/fridrich/Research/jpeg-phase-aware-Final.pdf)

In fact, the algorithm proposes a simple CNN architecture that was improved using of catalyst kernels as initialization to neurons and weight propagation via transfer learning.

The table below shows the results according to the detection error of each algorithm (`WOW`, `HUGO`, `S-UNIWARD`) with the payloads (`1.0`, `0.7` , `0.5` , `0.3`) bpp.

- Experiment result of **`S-UNIWARD`** :

| payload (bpp) | 1.0 | 0.7 | 0.5 | 0.3 |
| --- | --- | --- | --- | --- |
| Pe  | 0.11 | 0.22 | 0.31 | 0.34 |

- Experiment result of **`WOW`** :

| payload (bpp) | 1.0 | 0.7 | 0.5 | 0.3 |
| --- | --- | --- | --- | --- |
| Pe  | 0.08 | 0.17 | 0.21 | 0.30 |


- Experiment result of **`HUGO`** :

| payload (bpp) | 1.0 | 0.7 | 0.5 | 0.3 |
| --- | --- | --- | --- | --- |
| Pe  | 0.10 | 0.17 | 0.22 | 0.36 |


![alt text](images/detection_error_compare.png "Detection errors")

![alt text](images/model_roc_curves.png "ROC curves")

The paper of the article will be available soon.