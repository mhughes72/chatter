import collections
import cPickle
import numpy as np
from pkg_resources import resource_filename, Requirement

import ProcessorChain
import processors
import nlp
import dl
import topicxtract_structs

import tensorflow as tf

from util import constants


class TopicAnalyzer( ):
    def __init__(self, config=None):
        self.model_long = dl.model84_long( )
        self.model_short = dl.model84_short( )
        self.graph = tf.get_default_graph( )

        # load short answer data files
        if __name__ != "__main__":
            path = resource_filename("topicxtract_api", "DATA/long_vocab.pkl")
        else:
            path = "DATA/long_vocab.pkl"
        with open(path, "rb") as f:
            self.long_vocab = cPickle.load(f)

        if __name__ != "__main__":
            path = resource_filename("topicxtract_api", "DATA/long_labels.pkl")
        else:
            path = "DATA/long_labels.pkl"
        with open(path, "rb") as f:
            self.long_labels = cPickle.load(f)

        if __name__ != "__main__":
            path = resource_filename("topicxtract_api", "DATA/long_weights.hdf5")
        else:
            path = "DATA/long_weights.hdf5"
        self.model_long.load_weights(path)

        # load long answer data files
        if __name__ != "__main__":
            path = resource_filename("topicxtract_api", "DATA/short_vocab.pkl")
        else:
            path = "DATA/short_vocab.pkl"
        with open(path, "rb") as f:
            self.short_vocab = cPickle.load(f)

        if __name__ != "__main__":
            path = resource_filename("topicxtract_api", "DATA/short_labels.pkl")
        else:
            path = "DATA/short_labels.pkl"
        with open(path, "rb") as f:
            self.short_labels = cPickle.load(f)

        if __name__ != "__main__":
            path = resource_filename("topicxtract_api", "DATA/short_weights.hdf5")
        else:
            path = "DATA/short_weights.hdf5"
        self.model_short.load_weights(path)

        # init processor chain system.
        self.processor_chain = ProcessorChain.Chain( )
        self.processor_chain.append(processors.Sanitize)
        self.processor_chain.append(processors.Lemmatize)
        self.processor_chain.append(processors.Demojize)
        

    def analyze(self, answers):
        if not isinstance(answers, list):
            raise Exception("TopicAnalyzer.analyze answers must be input as list!!! Perhaps you are looking for TopicAnalyzer.analyze_one?")

        long_answers = [ ]
        short_answers = [ ]
        zip_array = [ ]
        for answer in answers:
            if not isinstance(answer, topicxtract_structs.Answer):
                raise Exception("Please use TopicAnalyzer.make_answer to create an Answer NamedTuple for use in the analyze functions!")
            
            split_reply = answer.reply.split(" ")
            concat_answer = answer.question + " " + answer.reply
            if len(split_reply) > constants.ANSWER_SPLIT_LENGTH:
                # long answer
                long_answers.append(concat_answer)
                zip_array.append({'kind':"long", 'idx':len(long_answers)-1})
            else:
                # short answer
                short_answers.append(concat_answer)
                zip_array.append({'kind':"short", 'idx':len(short_answers)-1})

        # prepare long/short
        long_sentences = self.processor_chain.run(long_answers)
        short_sentences = self.processor_chain.run(short_answers)

        long_sent_vectors = nlp.sent2vec(long_sentences, self.long_vocab)
        short_sent_vectors = nlp.sent2vec(short_sentences, self.short_vocab)

        long_raw_predictions = [ ]
        short_raw_predictions = [ ]
        with self.graph.as_default():
                long_raw_predictions = self.model_long.predict(long_sent_vectors)
                short_raw_predictions = self.model_short.predict(short_sent_vectors) # run machine learning.

        # make output.
        long_predictions = []
        for answer_predictions in long_raw_predictions:
            temp = {}
            for i, prediction_pct in enumerate(answer_predictions):
                temp[self.long_labels[i]] = prediction_pct
            long_predictions.append(temp)

        short_predictions = []
        for answer_predictions in short_raw_predictions:
            temp = {}
            for i, prediction_pct in enumerate(answer_predictions):
                temp[self.short_labels[i]] = prediction_pct
            short_predictions.append(temp)

        predictions = [ ]
        for z in zip_array:
            if z['kind'] == "long":
                predictions.append(long_predictions[z['idx']])
            if z['kind'] == "short":
                predictions.append(short_predictions[z['idx']])

        return predictions, answers


    def analyze_one(self, answer):
        predictions, answers = self.analyze([answer])

        return predictions[0], answers[0]


    def get_label_dictionary_long(self):
        return self.long_labels


    def get_label_dictionary_short(self):
        return self.short_labels


    def make_answer(self, question, reply):
        # make answer struct.
        answer = topicxtract_structs.Answer(question=question,reply=reply)
        return answer


if __name__ == "__main__":
    # analyzer = TopicAnalyzer( )

    # label_list_long = analyzer.get_label_dictionary_long( )
    # print label_list_long

    # answer1 = analyzer.make_answer("What about our shop keeps you coming back?", "unique gifts")
    # print analyzer.analyze_one(answer1);
    # print

    # answer2 = analyzer.make_answer("What do you think about the products we carry?", "the brika booth at dx3 was by far my favourite! it was perfectly set up and it beautifully displayed all of the merchandise! catherine cramer was a breat")
    # print analyzer.analyze_one(answer2);
    # print

    # answer_list = [
    #     analyzer.make_answer("Please complete the following sentence: The one area I think LUSH could improve on, would be...","Too much customer service. Too much barging in on people who have JUST walked in"),
    #     analyzer.make_answer("Please complete the following sentence: The one area I think LUSH could improve on, would be...","making the signs a bit more clear."),
    #     analyzer.make_answer("Please complete the following sentence: The one area I think LUSH could improve on, would be...","Cost of bathbombs"),
    #     analyzer.make_answer("Please complete the following sentence: The one area I think LUSH could improve on, would be...","Bigger stores! They are always full and I think if they were bigger and things were more spaced out, the experience would be more enjoyable!"),
    #     analyzer.make_answer("Last question. Can you recall a member of our team that really stood out during your visit? If so, please explain.","very helpful tall blonde male worker dancing"),
    #     analyzer.make_answer("Last question. Can you recall a member of our team that really stood out during your visit? If so, please explain.","2 of them- there was a woman who greeted me when I entered the store and showed me all the awesome products, and my cashier, a male, gave me the great company background"),
    #     analyzer.make_answer(u"Excellent - Welcome back! :) How long has it been since your last visit? 1= Less than a week (excluding today) 2= A few weeks 3= A few months 4= More than 6 months 5= It's been years","yesterday was my first day"),
    # ]

    # topic_list = [
    #     "Store Experience, Service",
    #     "Store Experience",
    #     "Pricing",
    #     "Store Experience",
    #     "Service",
    #     "Service",
    #     "Previous Visit",
    # ]

    # preds, answers = analyzer.analyze(answer_list);
    # print
    # for i, pred in enumerate(preds):
    #     print pred
    #     print answers[i]
    #     print answer_list[i]
    #     print topic_list[i]
    #     print "*"*30




