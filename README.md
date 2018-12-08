TOPIC XTRACT API
================
- This is an API connector the work AIFOUNDED has done to extract topics from text survey messages.



Setup
============
#### Requires: ####
* g++
* build_essential ( ubuntu ), or build tools for your distribution.
* Python 2.7
* pip
* setuptools

#### Install Dependencies: ####
```
on ubuntu:
sudo apt-get install build-essential g++ python python-pip python-setuptools
pip install numpy nltk scipy sklearn emoji keras tensorflow
```

#### Install Module: ####
```
sudo python setup.py install
```



Usage
============

#### Import: ####
```
from topicxtract_api import TopicAnalyzer
# Note: If you are getting the error: "No Module Named: topicxtract_api"
# please type which python, and run the "Install Module" command again with the output.
# it is likely the module is installing under the wrong python.
```



#### Initialize: ####
```
analyzer = TopicAnalyzer( )
```


#### Print Label Options: ####
````
label_list_long = analyzer.get_label_dictionary_long( )
label_list_short = analyzer.get_label_dictionary_short( )
print label_list_long
# ['' 'Campaign Idea' 'Charity' 'Company' 'Email ID' 'Hardware' 'Human Resources' 'No Suggestions' 'Phone Agent' 'Previous Visit' 'Pricing' 'Product' 'Purpose of Items Bought' 'Reason for Visit' 'Service' 'Store Experience']
````


#### Build An Answer Object ####
```
answer = analyzer.make_answer(QUESTION, REPLY_ANSWER)
```

```
Example: answer = analyzer.make_answer("What about our shop keeps you coming back?", "unique gifts")
```


#### Analyze An Answer: ####
```
prediction, answer = analyzer.analyze_one(answer)
```


* prediction: This is the prediction matrix as dict format label:prediction_percent:
```
Example: {'': 0.00069760886, 'Reason for Visit': 0.0022512842, 'Product': 0.93177444, 'Phone Agent': 0.001008615, 'Service': 0.010697231, 'Purpose of Items Bought': 0.0009784292, 'Company': 0.0062848814, 'Store Experience': 0.020162463, 'Hardware': 0.0017012653, 'Previous Visit': 0.0075042234, 'Pricing': 0.0059583616, 'No Suggestions': 0.0073023494, 'Campaign Idea': 0.0033034978, 'Charity': 0.00037530815}
```
* answer: This is the answer that you input ( for convenience & testing )



#### Analyze Multiple Answers: ####
```
answer_list = [
    analyzer.make_answer("Please complete the following sentence: The one area I think LUSH could improve on, would be...","Too much customer service. Too much barging in on people who have JUST walked in"),
    analyzer.make_answer("Please complete the following sentence: The one area I think LUSH could improve on, would be...","making the signs a bit more clear."),
    analyzer.make_answer("Please complete the following sentence: The one area I think LUSH could improve on, would be...","Cost of bathbombs"),
    analyzer.make_answer("Please complete the following sentence: The one area I think LUSH could improve on, would be...","Bigger stores! They are always full and I think if they were bigger and things were more spaced out, the experience would be more enjoyable!"),
    analyzer.make_answer("Last question. Can you recall a member of our team that really stood out during your visit? If so, please explain.","very helpful tall blonde male worker dancing"),
    analyzer.make_answer("Last question. Can you recall a member of our team that really stood out during your visit? If so, please explain.","2 of them- there was a woman who greeted me when I entered the store and showed me all the awesome products, and my cashier, a male, gave me the great company background"),
    analyzer.make_answer(u"Excellent - Welcome back! :) How long has it been since your last visit? 1= Less than a week (excluding today) 2= A few weeks 3= A few months 4= More than 6 months 5= It's been years","yesterday was my first day"),
]

predictions, answers = analyzer.analyze(answer_list);
```


* predictions: This is an array of prediction matrixes:
```
Example: [mat0, mat1, mat2, ...]
```
* answers: This is the answer array that you input ( for convenience & testing ):
```
Example: [a0, a1, a2, ...]
```
