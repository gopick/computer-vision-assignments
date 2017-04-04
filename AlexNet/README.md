# AlexNet

Alexnet implemented using tensorflow which also has timing information. 
Code for alexnet is obtained from _http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/_
I have added the timing information

##Running alexnet
Download the weights from the above website and put them in the working directory and use the command

```
python alexnet_forward.py
```
_timeline.json_ will be generated

load this file through chrome at chrome://tracing
