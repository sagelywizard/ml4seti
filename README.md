# ml4seti

This repository contains my entry in the ml4seti signal classification competition.

Example usage:

```
python main.py /path/to/ml4seti/full/ -i /path/to/index_file.csv --cuda --save --train --test 
```

The first argument is the path to the simulation `.dat` files. In addition, one must provide the path to the index file, which is a CSV of the file UUIDs and their classes (for training data). See the [ML4SETI Get Data notebook](https://github.com/setiQuest/ML4SETI/blob/master/tutorials/Step_1_Get_Data.ipynb).

Note that this is written in Python 3.x. The default ibmseti package (1.0.5) only supports Python 2.x.

A pre-released version of ibmseti runs on Python 3.5 and can be installed with `pip install ibmseti==2.0.0.dev5`.
