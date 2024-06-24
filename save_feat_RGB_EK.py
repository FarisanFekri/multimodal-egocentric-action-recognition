import pickle
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks

# global variables among training functions
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Feature Extraction")
    logger.info("Running with parameters: " + pformat_dict(args, indent=1)) #this takes it from configs/default.yaml and configs/I3D_save_feat.yaml


    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


def main():
    global modalities
    init_operations()
    modalities = args.modality

    # recover valid paths, domains, classes
    # this will output the domain conversion (D1 -> 8, et cetera) and the label list
    #so for D1-D1 .> source_domain =8, target_domain = 8
    num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
    #num_classes, valid_labels, source_domain, target_domain =  8 [0, 1, 2, 3, 4, 5, 6, 7] 8 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    train_augmentations = {}
    test_augmentations = {}
    logger.info("Instantiating models per modality")
    for m in modalities:
        
        #num_classes = 8
        #i found all of these in I3D_save_feat.yaml
        #m is 'RGB' 
        #args.models[m].model is I3D 
        #kwargs: {}
        #train_augmentations[m], test_augmentations[m] are just augmentations of I3D in aml23-ego/models/I3D.py
        
        logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))
        models[m] = getattr(model_list, args.models[m].model)(num_classes, m, args.models[m], **args.models[m].kwargs)
        # so then models contains RGB as key and model I3D instantianed as a key
        #models = {'RGB' : I3D } 
        
        train_augmentations[m], test_augmentations[m] = models[m].get_augmentation(m)

    action_classifier = tasks.ActionRecognition("action-classifier", models, 1,
                                                args.total_batch, args.models_dir, num_classes,
                                                args.save.num_clips, args.models, args=args)
    action_classifier.load_on_gpu(device)
    if args.resume_from is not None:
        action_classifier.load_last_model(args.resume_from)

    if args.action == "save":
        augmentations = {"train": train_augmentations, "test": test_augmentations}
        # the only action possible with this script is "save"
        #print(" modalities, args.split, args.dataset,args.save.num_frames_per_clip, args.save.num_clips, args.save.dense_sampling,augmentations[args.split]", modalities,args.split, args.dataset,args.save.num_frames_per_clip,args.save.num_clips, args.save.dense_sampling,augmentations[args.split])


        
        loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[1], modalities,
                                                                 args.split, args.dataset,
                                                                 args.save.num_frames_per_clip,
                                                                 args.save.num_clips, args.save.dense_sampling,
                                                                 augmentations[args.split], additional_info=True,
                                                                 **{"save": args.split}),
                                             batch_size=1, shuffle=False,
                                             num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes)
    else:
        raise NotImplementedError


def save_feat(model, loader, device, it, num_classes):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities

    model.reset_acc()
    model.train(False)
    results_dict = {"features": []}
    num_samples = 0
    logits = {}
    features = {}
    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label, video_name, uid) in enumerate(loader):
            # i_val 0
            #data ({'RGB': tensor([[[[-0.5686, -0.5686, -0.5608,  ...
            #label tensor([0]), 
            #video_name ('P08_09',)
            #uid tensor([13744]))

            label = label.to(device)

            for m in modalities: #m is 'RGB'
                #batch, _, height, width = (0,1,2,3) the dimensions (idk what 1 is here but it does not matter)
                batch, _, height, width = data[m].shape
                data[m] = data[m].reshape(batch, args.save.num_clips,
                                          args.save.num_frames_per_clip[m], -1, height, width)
                #now batch, n_clips_perVid, n_frames_perClip, idk-1, height, width = (0,1,2,3,4,5)
                data[m] = data[m].permute(1, 0, 3, 2, 4, 5)
                
                 #now n_clips_perVid, batch,  idk-1, n_frames_perClip, height, width = (0,1,2,3,4,5)

                
                logits[m] = torch.zeros((args.save.num_clips, batch, num_classes)).to(device) #tensor with dimensions (num_clips, batch, num_classes)
                features[m] = torch.zeros((args.save.num_clips, batch, model.task_models[m]  #tensor with dimensions (num_clips, batch, feat_dim)
                                           .module.feat_dim)).to(device)
                #print("logits, features ->",logits, features)
            clip = {}
            for i_c in range(args.save.num_clips):
                for m in modalities:
                    clip[m] = data[m][i_c].to(device) #clip["RGB"] = data["RGB"][0]
            #the code above gets all the clips in input

                output, feat = model(clip) #model represents the trained neural network model, and clip contains the input data for each modality for a single clip. The model processes this input data through its layers, computes the output logits (predictions) and features, and returns them.
                feat = feat["features"]
                for m in modalities:
                    logits[m][i_c] = output[m] #logits for current clip for the current modality
                    features[m][i_c] = feat[m] #features for current clip for the current modality
            for m in modalities:
                logits[m] = torch.mean(logits[m], dim=0) #It calculates the mean of the logits across all clips for that modality using torch.mean(logits[m], dim=0). This reduces the dimensionality of the logits tensor from (num_clips, batch_size, num_classes) to (batch_size, num_classes)
            for i in range(batch): #For each sample in the batch, creates a dictionary with keys uid and video_name
                sample = {"uid": int(uid[i].cpu().detach().numpy()), "video_name": video_name[i]}
                #print("uid,video_name of current sample", sample)
                for m in modalities:
                    sample["features_" + m] = features[m][:, i].cpu().detach().numpy()
                #so now
                #print("uid,video_nameOfSample, [features_RGB]", sample)
                #and results dict will contain all of those features for all the samples
                results_dict["features"].append(sample)
            num_samples += batch

            model.compute_accuracy(logits, label)

            if (i_val + 1) % (len(loader) // 5) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(loader),
                                                                          model.accuracy.avg[1], model.accuracy.avg[5]))

        os.makedirs("saved_features", exist_ok=True)
        pickle.dump(results_dict, open(os.path.join("saved_features", args.name + "_" +
                                                    args.dataset.shift.split("-")[1] + "_" +
                                                    args.split + ".pkl"), 'wb'))

        class_accuracies = [(x / y) * 100 for x, y in zip(model.accuracy.correct, model.accuracy.total)]
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1],
                                                                      model.accuracy.avg[5]))
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                         int(model.accuracy.correct[i_class]),
                                                         int(model.accuracy.total[i_class]),
                                                         class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies)}

    with open(os.path.join(args.log_dir, f'val_precision_{args.dataset.shift.split("-")[0]}-'
                                         f'{args.dataset.shift.split("-")[-1]}.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter, test_results['top1']))

    return test_results


if __name__ == '__main__':
    main()
