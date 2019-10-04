Steps to run the code.
1. Run "python cnn.py" to run the CNN Module which will display the accuracy obtained on the test data.
The code will automatically download the dataset and start running.
2. Run "python sketch_rnn.py" to run the RNN Module which will display the accuracy obtained for every 50 Epochs.
Stop this script whenever you see a satisfactory accuracy value.
Please ensure that your download the .npz files of the classes you want to use in model.py from this url.
https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn

Please note that the RNN code is obtained from the open source sketchRNN project. 
https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn

Their code is based on a generative model which generates sketches in realtime, we have removed both their decoder part and the loss function to add a classifier module using a cross entropy loss.
We also tried to use different loss functions, but due to some problems with the way they work, we left them commented for now and plan to work on it in the future.