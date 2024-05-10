#set page(numbering: "1")
#set par(justify: true)
#set text(size: 12pt)
#set figure(placement: auto)

#align(center, text(17pt)[
  *Indian Dance Forms Classification*
])

#let class_labels = (
  "bharatanatyam", "kathak", "kathakali", "kuchipudi", "manipuri", "mohiniyattam", "odissi", "sattriya", "purulia chhau",
)

= Introduction

Indian classical dance, or ICD, is a reflection of the country's rich cultural
legacy. Each dance originates from a distinct state within the nation. Nine
distinct dance forms are recognized by #link("https://www.indiaculture.gov.in/dance")
[Ministry of Culture, Government of India]. These dance forms include
Bharatanatyam, Kathak, Kathhakali, Kuchipudi, Manipuri, Odissi, Sattriya and
Chhau. Previous work in the paper “Classiﬁcation of Indian Dance Forms using
Pre-Trained Model-VGG” @biswas2021 was done for the eight dance forms excluding
Chhau and employed transfer learning to address this multiclass classification
in this project. Images of these eight dance forms that are known in India were
classified using pre-trained models like VGG16 and VGG19. The ICD dataset from
Kaggle, which included several photos in each of the eight classes was used.

= Literature Survey

In @samanta2012 the authors used a Computer Vision techniques and SVM to
classifiy Indian Classical Dance (ICD). They used a sparse representation based
dictionary learning technique. First, represent each frame of a dance video by a
pose descriptor based on histogram of oriented optical flow (HOOF), in a
hierarchical manner. The pose basis is learned using an on-line dictionary
learning technique. Finally each video is represented sparsely as a dance
descriptor by pooling pose descriptor of all the frames. In their work, dance
videos are classified using support vector machine (SVM) with intersection
kernel. An accuracy of 86.67% was achieved while tested their algorithm on their
own ICD dataset created from the videos collected from YouTube.

In @naik2020 the authors used Deep Learning Convolution Neural Network (CNN) to
classify five dance classes namely Bharatanatyam, Odissi, Kathak, Kathakali,
Yakshagana, the images of which are collected from the internet using Google
Crawler.

= Transfer Learning

Transfer learning is a machine learning technique where knowledge learned from a
task is re-used to boost performance on a related task. It is a popular approach
in deep learning as it enables the training of deep neural networks with less
data compared to having to create a model from scratch. The process involves
unfreezing some layers of the model and then retraining it at a low-learning
rate to handle a new dataset. Transfer learning is not a distinct type of
machine learning algorithm, but rather a technique or method used whilst
training models. By harnessing the ability to reuse existing models and their
knowledge of new problems, transfer learning has opened doors to training deep
neural networks even with limited data.

= VGG 16

VGG-16, or the Visual Geometry Group 16-layer model, is a deep convolutional
neural network architecture that was proposed by the Visual Geometry Group at
the University of Oxford. It was introduced in the paper titled "Very Deep
Convolutional Networks for Large-Scale Image Recognition" by K. Simonyan and A.
Zisserman in 2014 @simonyan2015deep. VGG-16 is a part of the VGG family of
models, which also includes VGG-19, VGG-M, and others.

== Architecture

#figure(image("screenshots/vgg16.png"), caption: "VGG-16 architecture Map")

The VGG-16 architecture is characterized by its simplicity and uniformity. It
consists of 16 weight layers, including 13 convolutional layers and 3 fully
connected layers. The convolutional layers are followed by max-pooling layers,
and the fully connected layers are followed by a softmax activation function for
classification.

The general architecture is as follows:

- *Input Layer* (224x224x3): Accepts an RGB image with a resolution of 224x224
  pixels.

- *Convolutional Layers* (Conv):
  - The first two layers have 64 filters with a 3x3 kernel and a stride of 1.
  - The next two layers have 128 filters with a 3x3 kernel and a stride of 1.
  - The next three layers have 256 filters with a 3x3 kernel and a stride of 1.
  - The final three layers have 512 filters with a 3x3 kernel and a stride of 1.

- *Max-Pooling Layers* (MaxPool):
  - After every two convolutional layers, max-pooling with a 2x2 window and a stride
    of 2 is applied.
  - The goal of the layer of pooling is to sample based on the input dimension. This
    decreases by this activation the number of parameters.

- *Fully Connected Layers* (FC):
  - There are three fully connected layers, each with 4096 neurons.
  - The last layer has 1000 neurons, corresponding to the 1000 ImageNet classes.

- *Softmax Layer*
  - The final layer uses the softmax activation function for classification.

@vgg16_summary_table shows the output shape and number of parameters for each
layer used in VGG16 model.

== Key Features:

- *Uniform Filter Size*

  VGG-16 uses a small 3x3 filter size throughout the convolutional layers,
  providing a uniform receptive field.

- *Deep Network*

  With 16 layers, VGG-16 is considered a deep neural network. The depth
  contributes to the model's ability to learn hierarchical features.

- *Ease of Interpretability*

  The uniform structure and simplicity of VGG-16 make it easy to interpret and
  understand compared to more complex architectures.

#let vgg16_summary = csv("screenshots/vgg16_layers.csv", delimiter: "|")
#figure(
  caption: "Summary of VGG-16.", table(
    columns: 3, inset: 7pt, align: horizon, ..vgg16_summary.at(0).map(strong).flatten(), ..vgg16_summary.slice(1).flatten(),
  ),
) <vgg16_summary_table>

= Dataset

We obtained from Kaggle the ICD dataset @icd_dataset, which included 599 images
from 8 distinct classes: Manipuri, Bharatanatyam, Odissi, Kathakali, Kathak,
Sattriya, Kuchipudi, and Mohiniyattam.

Purulia Chhau, the ninth class of dance, has also been added. Since the majority
of the pictures of chhau that are available online are of the purulia form, we
only added this subdance form. With the addition of Purulia Chhau a total of 653
images are present in the dataset.

#figure(
  image("screenshots/sample_images.png"), caption: "Dance forms images from the datasets.",
)

#figure(
  table(
    columns: (auto, auto, auto), inset: 10pt, align: horizon, [], [*Class*], [*Images*], "0", "bharatanatyam", "73", "1", "kathak", "76", "2", "kathakali", "74", "3", "kuchipudi", "76", "4", "manipuri", "84", "5", "mohiniyattam", "70", "6", "odissi", "71", "7", "sattriya", "75", "8", "purulia chhau", "53",
  ), caption: "Number of images in each class",
)

#figure(
  image("screenshots/bar_classes.png"), caption: "Number of images in each class",
)

= Training

VGG-16 was pretrained on the ImageNet Large Scale Visual Recognition Challenge
(ILSVRC) dataset, a large dataset containing millions of labeled images across
thousands of classes.

First, the data will be divided into training, testing, and validation samples.
The VGG16 model will then be trained using our training set of data. A portion
of it will be utilized at the validation stage. Next, using testing samples, we
will test the model.

#let training_csv = csv("screenshots/training.csv")

#figure(
  caption: "Training VGG16 model.", table(
    columns: 5, ..training_csv.at(0).map(strong).flatten(), ..training_csv.slice(1).flatten(),
  ),
)

= Performance Measurements

We further measured the performance of our model using a confusion matrix. The
main idea behind this tabular performance measurement is to understand how good
our model is when dealing with false-positives and false-negatives. Comparison
between the actual and predicted classes helped us to visualize the accuracy of
our model. In a confusion matrix, the true positive indicates the values which
are predicted correctly predicted as positive whereas the false positive refers
to the negative values which are incorrectly predicted as positive values.
Similarly, false negative tells us the number of positive values which are
predicted as negative by our model and lastly, the true negative refers to the
negative values which are correctly predicted as negative.

#figure(
  caption: "Confusion Matrix", table(
    columns: (auto, auto), inset: 10pt, align: horizon, [*True Positives (TP)*], [*False Negatives (FN)*], [*False Positives (FP)*], [*True Negatives (TN)*],
  ),
) <sample_confusion_matrix>

Accuracy is considered as one of the most important factors to measure the
performance. It is the ratio of correctly predicted observation to the total
number of observations. Mathematically,

$ "Accuracy" = ("TP" + "TN" ) / ("TP" + "FP" + "FN" + "TN" ) $

Next factor is precision. It refers to the ratio of correctly predicted positive
values to the total predicted positive values. Mathematically,

$ "Precision" = "TP" / ("TP" + "FP") $

Recall indicates to the ratio of correctly predicted positive observations to
the all observations in actual class. Mathemat- ically,

$ "Recall" = "TP" / ("TP" + "FN") $

Lastly, F1 Score is the weighted average of Precision and Recall.
Mathematically,

$ "F1 Score" = 2 * ("Recall" * "Precision") / ("Recall" + "Precision") $

@sample_confusion_matrix shows a sample confusion matrix for binary
classification. @classification_report shows the parameters of evaluation and
@confusion_matrix shows the confusion matrix for the VGG16 model trained and
tested on our dataset.

#{
  let class_report = csv("screenshots/classification_report.csv")
  let confusion_matrix = csv("screenshots/confusion_matrix.csv")

  let a = figure(
    caption: "Classification Report", table(
      columns: 5, inset: 10pt, align: horizon, ..class_report.at(0).map(strong).flatten(), ..class_report.slice(1).map(
        x => x.enumerate().map(y => { let (ind, val) = y; if ind == 0 { strong(val) } else { val } }),
      ).flatten(),
    ),
  )

  let b = figure(
    caption: "Confusion Matrix", table(
      columns: 10, inset: 10pt, align: horizon, ..confusion_matrix.at(0).map(strong).flatten(), ..confusion_matrix.slice(1).map(
        x => x.enumerate().map(y => { let (ind, val) = y; if ind == 0 { strong(val) } else { val } }),
      ).flatten(),
    ),
  )

  [ #a <classification_report> #b <confusion_matrix> ]
}

The number of epochs is a hyperparameter. We have implemented our models using
30 epochs for VGG16. The number of times the learning algorithm will work
through the entire training data is referred as epoch. We have plotted the
train_loss vs val_loss and train_acc vs val_acc for VGG 16 model.

#figure(
  image("screenshots/train_acc.png", width: 80%), caption: [Training accuracy vs validation accuracy for VGG-16.],
) <train_acc_16>

#figure(
  image("screenshots/train_loss.png", width: 80%), caption: [Training loss vs validation loss for VGG-16.],
) <train_loss_16>

@train_acc_16 shows the training accuracy vs validation accuracy for different
epochs. @train_loss_16 shows the training loss vs validation loss for different
epochs.

#pagebreak()
#bibliography("references.bib", title: "References")
