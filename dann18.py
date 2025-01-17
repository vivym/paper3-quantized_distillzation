import model_manager
import torch
import os
import datasets
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
import quantization
import pickle
import copy
import quantization.help_functions as qhf
import functools
import helpers.functions as mhf
from cnn_models.model import CNNModel
from cnn_models.model_small import SmallCNNModel


datasets.BASE_DATA_FOLDER = '...'
SAVED_MODELS_FOLDER = '...'
USE_CUDA = torch.cuda.is_available()

# cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
# print('CUDA_VISIBLE_DEVICES: {} for a total of {}'.format(cuda_devices, len(cuda_devices)))
# NUM_GPUS = len(cuda_devices)
NUM_GPUS = 1

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass


dann18Manager = model_manager.ModelManager('model_manager_dann18.tst',
                                            'model_manager', create_new_model_manager=True)
dann18modelsFolder = os.path.join(SAVED_MODELS_FOLDER, 'dann18_cnn')

for x in dann18Manager.list_models():
    if dann18Manager.get_num_training_runs(x) >= 1:
        s = '{}; Last prediction acc: {}, Best prediction acc: {}'.format(x,
                                            dann18Manager.load_metadata(x)[1]['predictionAccuracy'][-1],
                                            max(dann18Manager.load_metadata(x)[1]['predictionAccuracy']))
        print(s)
try:
    os.mkdir(dann18modelsFolder)
except:pass

epochsToTrain = 10

batch_size = 100
if batch_size % NUM_GPUS != 0:
    raise ValueError('Batch size: {} must be a multiple of the number of gpus:{}'.format(batch_size, NUM_GPUS))

train_loader, test_loader, original_loader = datasets.dataloader()
# train_loader, test_loader = dann18.getTrainLoader(batch_size), dann18.getTestLoader(batch_size)

TRAIN_TEACHER_MODEL = False
TRAIN_SMALLER_MODEL = True
TRAIN_DISTILLED_MODEL = True
TRAIN_DISTILLED_QUANTIZED_MODEL = True
CHECK_PM_QUANTIZATION = True

# Teacher model
teacher_model_name = 'dann18_teacher'
teacherModelPath = os.path.join(dann18modelsFolder, teacher_model_name)
teacherModel = CNNModel()
if USE_CUDA: teacherModel = teacherModel.cuda()
if NUM_GPUS > 1:
    teacherModel = torch.nn.parallel.DataParallel(teacherModel)

if teacher_model_name not in dann18Manager.saved_models:
    dann18Manager.add_new_model(teacher_model_name, teacherModelPath)
if TRAIN_TEACHER_MODEL:
    dann18Manager.train_model(teacherModel, model_name=teacher_model_name,
                               train_function=convForwModel.train_model,
                               arguments_train_function={'epochs_to_train': epochsToTrain,
                                                         'initial_learning_rate':0.01,
                                                         'print_every':50,
                                                         'learning_rate_style':'cifar100',
                                                         'weight_decayL2': 0.0005},
                               train_loader=train_loader, test_loader=test_loader)

teacherModel.load_state_dict(torch.load("./DANN18/models/mnist_mnistm_model_epoch_current_dict.pth"))


# smaller and distilled

smaller_model_name = 'dann18_smaller_model'
smaller_model_path = os.path.join(dann18modelsFolder, smaller_model_name)
smallerModel = SmallCNNModel()
if USE_CUDA: smallerModel = smallerModel.cuda()
if NUM_GPUS > 1: smallerModel = torch.nn.parallel.DataParallel(smallerModel)
if not smaller_model_name in dann18Manager.saved_models:
    dann18Manager.add_new_model(smaller_model_name, smaller_model_path)
if TRAIN_SMALLER_MODEL:
    dann18Manager.train_model(smallerModel, model_name=smaller_model_name,
                               train_function=convForwModel.train_model,
                               arguments_train_function={'epochs_to_train': epochsToTrain,
                                                         'print_every':50,
                                                         'initial_learning_rate':0.01,
                                                         'learning_rate_style':'cifar100',
                                                         'weight_decayL2':0.0005},
                               train_loader=train_loader, test_loader=test_loader)
#smallerModel.load_state_dict(cifar10Manager.load_model_state_dict(smaller_model_name))
del smallerModel

distilled_model_name = 'dann18_distilled_model'
distilled_model_path = os.path.join(dann18modelsFolder, distilled_model_name)
distilledModel = SmallCNNModel()
if USE_CUDA: distilledModel = distilledModel.cuda()
if NUM_GPUS > 1: distilledModel = torch.nn.parallel.DataParallel(distilledModel)
if not distilled_model_name in dann18Manager.saved_models:
    dann18Manager.add_new_model(distilled_model_name, distilled_model_path)
if TRAIN_DISTILLED_MODEL:
    dann18Manager.train_model(distilledModel, model_name=distilled_model_name,
                               train_function=convForwModel.train_model,
                               arguments_train_function={'epochs_to_train': epochsToTrain,
                                                         'print_every':50,
                                                         'initial_learning_rate':0.01,
                                                         'learning_rate_style':'cifar100',
                                                         'weight_decayL2':0.0005,
                                                         'teacher_model': teacherModel,
                                                         'use_distillation_loss': True},
                               train_loader=train_loader, test_loader=test_loader)
#distilledModel.load_state_dict(cifar10Manager.load_model_state_dict(distilled_model_name))
del distilledModel

numBits = [2, 4]
for numBit in numBits:
    distilled_quantized_model_name = 'dann18_distilled_quantized{}bits'.format(numBit)

    distilled_quantized_model_path = os.path.join(dann18modelsFolder, distilled_quantized_model_name)
    distilled_quantized_model = SmallCNNModel()
    if USE_CUDA: distilled_quantized_model = distilled_quantized_model.cuda()
    if NUM_GPUS > 1: distilled_quantized_model = torch.nn.parallel.DataParallel(distilled_quantized_model)
    if not distilled_quantized_model_name in dann18Manager.saved_models:
        dann18Manager.add_new_model(distilled_quantized_model_name, distilled_quantized_model_path)
    if TRAIN_DISTILLED_QUANTIZED_MODEL:
        dann18Manager.train_model(distilled_quantized_model, model_name=distilled_quantized_model_name,
                                   train_function=convForwModel.train_model,
                                   arguments_train_function={'epochs_to_train': epochsToTrain,
                                                             'teacher_model': teacherModel,
                                                             'use_distillation_loss': True,
                                                             'quantizeWeights':True,
                                                             'numBits':numBit,
                                                             'bucket_size':256,
                                                             'print_every':50,
                                                             'initial_learning_rate':0.01,
                                                             'learning_rate_style':'cifar100',
                                                             'weight_decayL2':0.0005,
                                                             'quantize_first_and_last_layer':False},
                                   train_loader=train_loader, test_loader=test_loader)
    #distilled_quantized_model.load_state_dict(cifar10Manager.load_model_state_dict(distilled_quantized_model_name))
    del distilled_quantized_model

del teacherModel

def load_model_from_name(x):
    # opt = dann18Manager.load_metadata(x, 0)[0]
    # #small old bug in the saving of metadata, this is a cheap trick to remedy it
    # for key, val in opt.items():
    #     if isinstance(val, str):
    #         opt[key] = eval(val)
    model = SmallCNNModel()
    if USE_CUDA: model = model.cuda()
    try:
        model.load_state_dict(dann18Manager.load_model_state_dict(x))
    except:
        model.load_state_dict(mhf.convert_state_dict_from_data_parallel(
            dann18Manager.load_model_state_dict(x)))
    return model

for x in dann18Manager.list_models():
    if dann18Manager.get_num_training_runs(x) == 0:
        continue
    model = load_model_from_name(x)
    reported_accuracy = dann18Manager.load_metadata(x)[1]['predictionAccuracy'][-1]
    pred_accuracy = cnn_hf.evaluateModel(
        model, test_loader, fastEvaluation=False,
        confusion_matrix_path=f"results/{x}_cfm.png",
        tsne_path=f"results/{x}_tsne.png",
    )
    print('Model "{}" ==> Prediction accuracy: {:2f}% == Reported accuracy: {:2f}%'.format(x,
                                                        pred_accuracy*100, reported_accuracy*100))
    curr_num_bit = dann18Manager.load_metadata(x)[0].get('numBits', None)
    if curr_num_bit is not None:
        quant_fun = functools.partial(quantization.uniformQuantization, s=2**curr_num_bit, bucket_size=256)
        actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(model.parameters(), quant_fun,
                                                                       'uniform', s=2**curr_num_bit)
        size_reduction = mhf.get_size_reduction(actual_bit_huffmman, bucket_size=256)
        size_model_MB = mhf.get_size_quantized_model(model, curr_num_bit, quant_fun,
                                                             bucket_size=256, quantizeFirstLastLayer=False)
        print('Effective bit Huffman: {} - Size reduction: {} - Size MB : {}'.format(actual_bit_huffmman,
                                                                                     size_reduction, size_model_MB))
    else:
        size_model_MB = mhf.getNumberOfParameters(model)*4/1000000
        print('Size MB : {}'.format(size_model_MB))

    if CHECK_PM_QUANTIZATION:
        if 'distilled' in x and 'quant' not in x:
            for numBit in numBits:
                try:
                    model.load_state_dict(dann18Manager.load_model_state_dict(x))
                except:
                    model.load_state_dict(mhf.convert_state_dict_from_data_parallel(
                        dann18Manager.load_model_state_dict(x)))
                numParam = sum(1 for _ in model.parameters())
                for idx, p in enumerate(model.parameters()):
                    if idx == 0 or idx == numParam - 1:
                        continue
                    p.data = quantization.uniformQuantization(p.data, s=2**numBit, type_of_scaling='linear',
                                                              bucket_size=256)[0]
                predAcc = cnn_hf.evaluateModel(
                    model, test_loader, fastEvaluation=False,
                    confusion_matrix_path=f"results/{x}_cfm_check_pm.png",
                    tsne_path=f"results/{x}_tsne_check_pm.png",
                )
                # predAcc =0
                print('PM quantization of model "{}" with "{}" bits and bucketing 256: {:2f}%'.format(x, numBit, predAcc * 100))
                quant_fun = functools.partial(quantization.uniformQuantization, s=2**numBit, bucket_size=None)
                actual_bit_huffmman = qhf.get_huffman_encoding_mean_bit_length(model.parameters(), quant_fun,
                                                                               'uniform',s=2**numBit)
                size_reduction = mhf.get_size_reduction(actual_bit_huffmman, bucket_size=256)
                size_model_MB = mhf.get_size_quantized_model(model, numBit, quant_fun,
                                                             bucket_size=256, quantizeFirstLastLayer=False)
                print('Effective bit Huffman: {} - Size reduction: {} - Size MB: {}'.format(
                                                                            actual_bit_huffmman,
                                                                            size_reduction,
                                                                            size_model_MB))
