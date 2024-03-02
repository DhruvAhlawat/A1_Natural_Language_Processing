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