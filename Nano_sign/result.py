epoch 50

Loading and preparing results...
DONE (t=0.16s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.68s).
Accumulating evaluation results...
DONE (t=0.23s).
[NanoDet][01-23 17:49:59]INFO:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.689
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.903
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.796
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.350
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.595
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.790
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.703
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.774
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.774
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.717
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.861

[NanoDet][01-23 17:49:59]INFO:
| class               | AP50   | mAP   | class                  | AP50   | mAP   |
|:--------------------|:-------|:------|:-----------------------|:-------|:------|
| Keep left           | 62.2   | 43.4  | Keep right             | 86.4   | 62.7  |
| No U-turn           | 92.4   | 77.5  | No left turn           | 73.8   | 58.6  |
| No parking          | 95.1   | 72.5  | No right turn          | 67.7   | 55.9  |
| No stopping         | 96.4   | 76.5  | Parking                | 92.1   | 66.6  |
| Pedestrian Crossing | 98.7   | 69.2  | Speed Limit -100-      | 99.5   | 86.1  |
| Speed Limit -30-    | 99.6   | 90.0  | Speed Limit -60-       | 99.3   | 89.9  |
| Stop Sign           | 95.5   | 89.3  | Traffic Light -Green-  | 87.3   | 55.6  |
| Traffic Light -Red- | 94.5   | 58.9  | Traffic Light -Yellow- | 94.5   | 73.3  |
| U-turn              | 98.4   | 82.2  | bike                   | 82.5   | 49.5  |
| motobike            | 93.1   | 65.6  | person                 | 93.8   | 57.3  |
| vehicle             | 92.9   | 66.8  |                        |        |       |
[NanoDet][01-23 17:50:00]INFO:Saving model to /home/newin/Projects/nanodet_sign/workspace/traffic_sign/model_best/nanodet_model_best.pth
[NanoDet][01-23 17:50:00]INFO:Val_metrics: {'mAP': 0.6893883586425867, 'AP_50': 0.9027360610072815, 'AP_75': 0.795960883487729, 'AP_small': 0.34998578931051755, 'AP_m': 0.5945055185152328, 'AP_l': 0.7898535697698943}
`Trainer.fit` stopped: `max_epochs=50` reached.
(nanoenv_gpu) newin@newinloq:~/Projects/nanodet_sign/nanodet$ 
