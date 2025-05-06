## Notable method 

The most notable method I used to improve my accuracies can essentially be explained as a form of label smoothing. Since it was quite clear that the training data had several incorrect labels as real datasets usually have, I did not just train on the original labels. after training my model once, since the accuracy was high ( 91-93%), I used this same model to relabel all the training data, then I increased the stored Naive Bayes weights by a factor of α which was around 1.1, before continuing training again on this modified dataset, (α > 1 so there is higher weights to the original training data). I did this process for a number of iterations and fine-tuned the value of α to get the highest final accuracy over the validation set. This increased my accuracy to about 95% on the val set, almost a 2-3% increase.
 
 Note: I did not train the Naive Bayes model again, sklearn also has a feature
 that lets you ”continue” the training and that is what I did. Hence I manipulated
 its stored counts first before training further.

## Steps to run:

1. To train on the data stored in directory "data" and save the model in the "savedir" directory, run the following:

```bash
bash run_model.sh train data savedir
```
2. To test the model on the data stored in directory "data" and the model in the "savedir" directory, run the following:

```bash
bash run_model.sh test savedir data outfilename
```
this outputs a file of predictions named "outfilename"
