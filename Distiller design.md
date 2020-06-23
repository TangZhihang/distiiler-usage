
#蒸馏器设计<br>
Distiller旨在轻松集成到您自己的PyTorch研究应用程序中。
通过检查用于压缩图像分类模型的示例应用程序的代码（compress_classifier.py），最容易理解这种集成。

该应用程序从Torchvision的[ImageNet分类培训样本应用程序](https://github.com/pytorch/examples/tree/master/imagenet "标题")借用了其主要流程代码。 我们试图使其相似，以使其熟悉并易于理解。

集成压缩非常简单：对于培训的每个阶段，只需添加适当的compression_scheduler回调的调用即可。 训练框架看起来像下面的伪代码。 样板式Pytorch分类培训与CompressionScheduler的调用有关。

 ```textmate
For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    save_checkpoint()
    compression_scheduler.on_epoch_end(epoch)

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input_var)
        loss = criterion(output, target_var)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)



```
这些回调可以在下图中看到，如从Training Loop指向Distiller的Scheduler的箭头，该调用正确的算法。 该应用程序还使用Distiller服务来收集摘要和日志文件中的统计信息，以后可以从Jupyter notebook 或TensorBoard中查询这些文件。



![流程](.\img\distiller-design.png)

#稀疏和微调
* 该应用程序按照通常在PyTorch中进行的设置模型。
* 然后实例化一个Scheduler并对其进行配置：
    + 计划程序配置在YAML文件中定义
    + 该配置指定策略。每个策略都与控制训练某些方面的特定算法相关。
        - 某些类型的算法控制模型的实际稀疏性。这样的类型是“修剪器”和“调节器”。
        - 一些算法控制训练过程的某些参数，例如学习速率衰减调度程序（lr_scheduler）。
        - 每种算法的参数也在配置中指定。
* 除了指定算法之外，每个策略还指定调度参数，这些参数控制执行算法的时间：开始时期，结束时期和频率。
调度程序公开了有关训练阶段的回调：时期开始/结束，小批量开始/结束和预向后传递。每个调度程序回调都会激活根据定义的计划定义的策略。
* 这些回调放置在训练循环中。

#量化
通过将现有操作替换为量化版本来获得量化模型。 量化版本可以是完全替代版本，也可以是包装程序。 包装器将在内部使用现有模块，并在必要时在之前/之后添加量化和反量化操作。

在Distiller中，我们将提供一组常用操作的量化版本，以实现不同的量化方法。 用户可以使用提供的量化操作从头开始编写量化模型。

我们还提供一种机制，该机制采用现有模型，并自动用量化版本替换所需的操作。 该机制由Quantizer类公开。 对于每种量化方法，应将量化器分类。


##模型转换
高层流程如下：

定义要替换的模块类型（例如Conv2D，线性等）到生成替换模块的函数之间的映射。映射在Quantizer类的replace_factory属性中定义。
遍历模型中定义的模块。对于每个模块，如果其类型在映射中，请调用替换生成函数。我们将现有模块传递给此函数以允许对其进行包装。
用功能返回的模块替换现有模块。重要的是要注意，模块的名称不会更改，因为这可能会破坏父模块的转发功能。
显然，不同的量化方法可以使用不同的量化运算。另外，不同的方法可以采用替换/包装现有模块的不同“策略”。例如，某些方法用另一种激活功能替换ReLU，而另一些保留它。因此，对于每种量化方法，将可能定义不同的映射。
Quantizer的每个子类都应使用适当的映射填充replace_factory字典属性。
要执行模型转换，请调用Quantizer实例的prepare_model函数。

##灵活的位宽
通过量化不同张量类型所使用的位数来量化```Quantizer```的每个实例。默认值为激活和权重。这些是```Quantizer```构造函数中的bits_activations，bits_weights和bits_bias参数。子类可以根据需要为其他张量类型定义位宽。
我们还希望能够覆盖某些项目中上面项目符号中提到的默认位数。这些可能是非常具体的层。但是，许多模型都是由包含多个模块的构建块（“容器”模块，例如Sequential）组成的，很可能我们希望覆盖整个块或跨不同块的某个模块的设置。使用此类构造块时，内部模块的名称通常遵循某种模式。
因此，为此，Quantizer还接受正则表达式到位数的映射。这允许用户使用特定名称覆盖特定图层，或者通过正则表达式覆盖一组图层。此映射通过构造函数中的overrides参数传递。
覆盖映射必须是collections.OrderedDict的实例（而不是简单的Python dict）。这样做是为了能够处理重叠的名称模式。
因此，例如，可以为一组图层定义某些替代参数，例如```'conv *'```，但还可以为该组中的特定图层定义不同的参数，例如'conv1'。
对模式进行热切评估-第一场比赛获胜。因此，更具体的模式必须先于广泛的模式。


###权重量化<br>
Quantizer类还提供了一个API，可以一次性量化所有图层的权重。 要使用它，param_quantization_fn属性需要指向一个接受张量和位数的函数。 在模型转换期间，Quantizer类将构建所有需要量化的模型参数及其位宽的列表。 然后，可以调用quantumize_params函数，该函数将迭代所有参数并使用params_quantization_fn对其进行量化。

###量化意识培训<br>
Quantizer类支持量化感知训练，即-在循环中进行量化训练。这需要处理几个流程/场景：

1. 维护权重的完整精度副本，如此处所述。通过在Quantizer构造函数中设置```train_with_fp_copy = True```启用此功能。在模型转换时，在每个具有应量化参数的模块中，都会添加一个新的torch.nn.Parameter，它将维护所需的参数全精度副本。请注意，这是就地完成的-不创建新模块。为此，我们最好不要对现有的PyTorch模块进行子分类。为了做到这一点，并通过权重量化功能确保适当的向后传播，我们采用了以下“技巧”：

    1. 现有的torch.nn.Parameter，例如重量，将替换为名为float_weight的torch.nn.Parameter。
    2. 为了保持模块的现有功能，我们然后在模块中注册一个原始名称为weights的缓冲区。
    3. 在训练过程中，float_weight将传递给param_quantization_fn，结果将存储在weight中。
2. 此外，某些量化方法可能会将额外的学习参数引入模型。例如，在PACT方法中，将活动限制为值α，这是每层的学习参数

为了支持这两种情况，Quantizer类还接受torch.optim.Optimizer的一个实例（通常这是其子类的一个实例）。量化器将根据对参数的更改来修改优化器。
