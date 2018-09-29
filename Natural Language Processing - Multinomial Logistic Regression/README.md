# Multinomial Logistic Regression

**Goal:** To implement a working **Natural Language Processing (NLP)** system, i.e., a
**mini Siri**, using multinomial logistic regression. The Algorithm is used to **extract flight information** 
from natural text.

**Dataset:** Airline Travel Information System (ATIS) data set

**Tags:** The tags are in **Begin-Inside-Outside (BIO) format**. Tags starting with B indicating the beginning of a piece
of information, tags beginning with I indicating a continuation of a previous type of information, and O
tags indicating words outside of any information chunk

**Implementation:**

1) Initialize all model parameters to 0.

2) Use stochastic gradient descent (SGD) to optimize the parameters for a multinomial logistic regression
model. The number of times SGD loops through all of the training data (num epoch) will be
specified as a command line flag. Set your learning rate as a constant n = 0:5.

3) Be able to select which one of two feature structures you will use in your logistic regression model
using an command line flag (see Section 2.3.4)

4) To resolve ties where multiple classes have the same likelihood, choose the label with the smaller
ASCII value (e.g., ’Aardvark’ is less than ’Apple’; and ’Apple’ is less than ’apple’]).

**Feature Structures:**

**Model 1:** p(y | word , parameters): This model defines a probability distribution over the current tag y using the
parameters and a feature vector based on only the current word. This model should be used
when <feature flag> is set to 1.
  
**Model 2:** p(y | word_t-1 , word_t, word_t+1 , parameters): This model defines a probability distribution over the current tag y
using the parameters and a feature vector based on the previous word ( word_t-1 ), the current word (word_t) ,
and the next word (word_t+1) in the sequence. This model should be used when <feature flag> is set
to 2.
