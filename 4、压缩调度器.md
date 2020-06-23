###Compression scheduler

在迭代修剪中，我们创建了一种修剪方案，该修剪方案指定如何修剪以及在修剪和培训阶段的每个阶段修剪什么。 这激励了CompressionScheduler的设计：它需要成为训练循环的一部分，并且能够制定和实施修剪，正则化和量化决策。 我们希望能够更改压缩时间表的细节，而无需触摸代码，并决定使用YAML作为该规范的容器。 我们发现，当我们在同一代码库上进行许多实验时，如果将差异与代码库分离，则维护所有这些实验会变得更加容易。 因此，我们增加了对学习速率衰减调度的调度程序支持，并且，我们希望自由地更改LR衰减策略而不更改代码。

###High level overview
让我们简要地讨论主要机制和抽象：调度规范由定义Pruner，Regularizer，Quantizer，LR-Scheduler和Policys实例的部分列表组成。


* 修剪器，正则化器和量化器非常相似：它们分别实现修剪/正则化/量化算法。
* LR调度器指定LR衰减算法。

这些定义了时间表的哪一部分。

该策略定义了计划的何时部分：在哪个时期开始应用Pruner / Regularizer / Quantizer / LR-decay，该时期结束，以及调用策略的频率（应用频率）。 策略还定义了它正在管理的Pruner / Regularizer / Quantizer / LR-decay的实例。
从YAML文件或从字典配置CompressionScheduler，但是您也可以从代码手动创建Policy，Pruner，Regularizer和Quantizer。

###Syntax through example
我们将使用alexnet.schedule_agp.yaml来解释一些用于配置Alexnet敏感度修剪的YAML语法。
```yaml
version: 1
pruners:
  my_pruner:
    class: 'SensitivityPruner'
    sensitivities:
      'features.module.0.weight': 0.25
      'features.module.3.weight': 0.35
      'features.module.6.weight': 0.40
      'features.module.8.weight': 0.45
      'features.module.10.weight': 0.55
      'classifier.1.weight': 0.875
      'classifier.4.weight': 0.875
      'classifier.6.weight': 0.625

lr_schedulers:
   pruning_lr:
     class: ExponentialLR
     gamma: 0.9

policies:
  - pruner:
      instance_name : 'my_pruner'
    starting_epoch: 0
    ending_epoch: 38
    frequency: 2

  - lr_scheduler:
      instance_name: pruning_lr
    starting_epoch: 24
    ending_epoch: 200
    frequency: 1
```

YAML语法只有一个版本，并且该版本号目前尚未验证。 但是，为了适应未来的发展，最好是让YAML解析器知道您正在使用版本1语法，以防出现版本2。

```yaml
version: 1
```
在pruners部分，我们定义了希望调度程序实例化和使用的pruners实例。
我们定义了一个名为my_pruner的pruner实例。我们将在策略部分引用这个实例。
然后我们列出每个权值张量的灵敏度乘子s。
您可以在本节中列出任意数量的修剪器，只要每个修剪器有一个惟一的名称即可。您可以在一个调度中使用多种类型的修剪器。
```yaml
pruners:
  my_pruner:
    class: 'SensitivityPruner'
    sensitivities:
      'features.module.0.weight': 0.25
      'features.module.3.weight': 0.35
      'features.module.6.weight': 0.40
      'features.module.8.weight': 0.45
      'features.module.10.weight': 0.55
      'classifier.1.weight': 0.875
      'classifier.4.weight': 0.875
      'classifier.6.weight': 0.6
```

接下来，我们希望在lr_schedulers部分中指定学习速率衰减调度。我们为这个实例分配一个名称:pruning_lr。与pruners部分一样，您可以使用任何名称，
只要所有lr -调度器都有一个惟一的名称。目前，只允许LR-scheduler的一个实例。LR-scheduler必须是PyTorch的[_LRScheduler](https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html) 的子类。
您可以使用torch.optim中定义的任何调度器。lr_scheduler([见这里](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) )。此外，我们还在Distiller中实现了一些额外的调度器([参见这里](https://github.com/NervanaSystems/distiller/blob/master/distiller/learning_rate.py) )。
关键字参数直接传递给LR-scheduler的构造函数，以便在添加新的LR-scheduler时添加到torch.optim。lr_scheduler，
可以在不更改应用程序代码的情况下使用它们。

```yaml
lr_schedulers:
   pruning_lr:
     class: ExponentialLR
     gamma: 0.9
```
最后，我们定义了policy部分，它定义了实际的调度。策略通过命名实例来管理一个Pruner、Regularizer、Quantizer或LRScheduler的实例。在下例中，PruningPolicy使用名为my_pruner的pruner实例:它以2个epoch(即每隔一个epoch)的频率激活它，从epoch 0开始，到epoch 38结束。
```yaml
policies:
  - pruner:
      instance_name : 'my_pruner'
    starting_epoch: 0
    ending_epoch: 38
    frequency: 2

  - lr_scheduler:
      instance_name: pruning_lr
    starting_epoch: 24
    ending_epoch: 200
    frequency: 1
```
这是迭代裁剪:
1. 训练连接
2. 裁剪连接
3. 再次训练权值
4. 转到2

[ Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) 这篇论文描述了它。

"Our method prunes redundant connections using a three-step method. First, we train the network to learn which connections are important. Next, we prune the unimportant connections. Finally, we retrain the network to fine tune the weights of the remaining connections...After an initial training phase, we remove all connections whose weight is lower than a threshold. This pruning converts a dense, fully-connected layer to a sparse layer. This first phase learns the topology of the networks — learning which connections are important and removing the unimportant connections. We then retrain the sparse network so the remaining connections can compensate for the connections that have been removed. The phases of pruning and retraining may be repeated iteratively to further reduce network complexity."


###Regularization

您还可以定义和安排正则化。
####L1正则化








###模型蒸馏

知识蒸馏(参见这里)也作为策略实现，应该添加到调度器中。但是，对于当前实现，不能像上面描述的其他策略那样在YAML文件中定义它。
为了使这种方法集成到应用程序中更容易一些，可以使用一个助手函数，它将添加一组与知识蒸馏相关的命令行参数:
```python
import argparse
import distiller

parser = argparse.ArgumentParser()
distiller.knowledge_distillation.add_distillation_args(parser)
```
(add_distillation_args函数接受一些可选参数，详细信息请参阅在distiller/ knowledge_distillib .py上的实现)<br>
下面是这个函数给出的命令行参数:

```shell script
Knowledge Distillation Training Arguments:
  --kd-teacher ARCH     Model architecture for teacher model
  --kd-pretrained       Use pre-trained model for teacher
  --kd-resume PATH      Path to checkpoint from which to load teacher weights
  --kd-temperature TEMP, --kd-temp TEMP
                        Knowledge distillation softmax temperature
  --kd-distill-wt WEIGHT, --kd-dw WEIGHT
                        Weight for distillation loss (student vs. teacher soft
                        targets)
  --kd-student-wt WEIGHT, --kd-sw WEIGHT
                        Weight for student vs. labels loss
  --kd-teacher-wt WEIGHT, --kd-tw WEIGHT
                        Weight for teacher vs. labels loss
  --kd-start-epoch EPOCH_NUM
                        Epoch from which to enable distillation
```
一旦参数被解析，一些初始化代码是必需的，类似如下:

```python
# Assuming:
# "args" variable holds command line arguments
# "model" variable holds the model we're going to train, that is - the student model
# "compression_scheduler" variable holds a CompressionScheduler instance
import args
args.kd_policy = None
if args.kd_teacher:
    # Create teacher model - replace this with your model creation code
    teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
    if args.kd_resume:
        teacher, _, _ = apputils.load_checkpoint(teacher, chkpt_file=args.kd_resume)

    # Create policy and add to scheduler
    dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
    args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
    compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                     frequency=1)
```
最后，在训练循环中，我们还需要通过教师模型进行正向传播。KnowledgeDistillationPolicy类保留对student和teacher模型的引用，并公开一个forward函数，该函数在这两个模型上执行前向传播。由于这不是一个标准的策略回调，我们需要从我们的训练循环手动调用这个函数，如下所示:

```python
if args.kd_policy is None:
    # Revert to a "normal" forward-prop call if no knowledge distillation policy is present
    output = model(input_var)
else:
    output = args.kd_policy.forward(input_var)
```

To see this integration in action, take a look at the image classification sample at examples/classifier_compression/compress_classifier.py.

要查看集成的实际效果，请查看示例/classifier_compression/compress_classifier.py中的图像分类示例。