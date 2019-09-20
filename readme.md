# Adversarial_MTER

This is the code of ICCV paper[《Adversarial Learning with Margin-based Triplet Embedding Regularization》](https://github.com/zhongyy/Adversarial_MTER/)

The Deep neural networks (DNNs) have achieved great success on a variety of computer vision tasks, however, they are highly vulnerable to adversarial attacks. To address this problem, we propose to improve the local smoothness of the representation space, by integrating a margin-based triplet embedding regularization term into the classification objective, so that the obtained model learns to resist adversarial examples. The two optimization steps in MTER, vulnerability exploitation and fixing, strive to attack each other but also together improve the robustness of DNNs gradually.

![arch](https://github.com/zhongyy/Adversarial_MTER/blob/master/illustration.jpg)

## Usage Instructions

The code is adopted from [InsightFace](https://github.com/deepinsight/insightface). I sincerely appreciate for their contributions.

1. Install `MXNet` with GPU support (Python 2.7).

```
pip install mxnet-cu90
```
2. download the code as `Adversarial_MTER/`
```
git clone https://github.com/zhongyy/Adversarial_MTER.git
```

3.  The CASIA-WebFace dataset is preprocessed by ourseleves using MTCNN. The VGGFace2 and MS1MV1 datasets are downloaded from Data Zoo of [InsightFace](https://github.com/deepinsight/insightface).  


4. MNIST models are trained from scratch.
```
CUDA_VISIBLE_DEVICES='0,1' python ad_mnist.py --loss-type 0 --verbose 30 --network o18 --lr 0.1 --lr-steps 8000,16000  --end-epoch 100 --main-per-batch-size 2700 --adv-per-batch-size 1800 --adv-round 50 --adv-alpha 0.20 --adv-sigma 0.35 --adv-thd 0.05 --weightadv 1 --prefix /home/zhongyaoyao/insightface/models/adv_mnist_o18_soft_0.20/model 2>&1|tee adv_mnist_o18_soft_0.20.log
```
```
CUDA_VISIBLE_DEVICES='0,1' python ad_mnist.py --loss-type 0 --verbose 30 --network o18 --lr 0.01 --lr-steps 8000,16000 --end-epoch 100 --main-per-batch-size 2700 --adv-per-batch-size 1800 --adv-round 100 --adv-alpha 0.20  --adv-sigma 0.35 --adv-thd 0.05 --weightadv 1 --prefix /home/zhongyaoyao/insightface/models/adv_mnist_o18_soft_1.20/model2 --pretrained /home/zhongyaoyao/insightface/models/adv_mnist_o18_soft_0.20/model,0 2>&1|tee adv_mnist_o18_soft_0.20_2.log
```

5. While face models are fintuned from pretrained [ArcFace models](https://github.com/deepinsight/insightface).  
Face models

```
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5' python -u ad_face.py --network r50  --loss-type 4 --margin-m 0.5  --data-dir /ssd/CASIA_rec/  --lr 0.01 --main-per-batch-size 90 --adv-per-batch-size 120 --ctx-num 4 --ctx-adv-num 2 --prefix ../models/ad_train/model --pretrained  /home/zhongyaoyao/insightface/models/r50_webface_arc/model,100 2>&1|tee adv_r50_webface_arc.log
```

