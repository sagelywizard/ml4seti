# ml4seti

This repository contains my entry in the ml4seti signal classification competition.

Example usage:

```
python main.py /path/to/ml4seti/full/ --cuda --save
```

Note that this is written in Python 3.x. The ibmseti package only supports Python 2.x, so I had to make a few changes to get the library to work with Python 3.x. The tweaked ibmseti library can be found [here](https://github.com/sagelywizard/ibmseti/tree/python3).
