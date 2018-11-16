# Lab2 - Multilyaer Perceptron

## File info
- `/Dataset`
    - Store all dataset
- `perceptron.py`
    - As the base 
    - Simulate the operation of perceptron
- `binaryClass.py` 
    - Just for practice
    - Using multilayer perceptron to deal with binary classification
    - Using all data as training data
- `multiClass.py`
    - Just for practice
    - Using multilayer perceptron to deal with multi classification
    - Using all data as training data
- `multiClassifier.py` 
    -  Logical part of my Lab2, simulate multilayer perceptron
    -  Able to deal with both multi or binary classification
- `panel.py`
    - UI part of my Lab2 by using PyQt
    - Program start from here, has `main()`
- `support.py`
    - Also as part of my Lab2
    - Using it to do operations which is just for supporting UI

## Target
- Binary Classification
- Multi Classification
- Deal with multi dimension
- Draw the figure for 2-dimension instances
    - Using another thread for drawing
    - Avoid from blocking main UI
- Compute recognition for training and testing data
- Compute RMSE of training
- Optimize with **Momentum**

## Mind
- If number of data < 20:
    - Treat all data as training data