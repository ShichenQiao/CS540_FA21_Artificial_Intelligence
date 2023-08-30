import os
import math


# These first two functions require os operations and so are completed for you
# Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d + "/"
        files = os.listdir(directory + subdir)
        for f in files:
            bow = create_bow(vocab, directory + subdir + f)
            dataset.append({'label': label, 'bow': bow})
    return dataset


# Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        subdir = d if d[-1] == '/' else d + '/'
        files = os.listdir(directory + subdir)
        for f in files:
            with open(directory + subdir + f, 'r') as doc:
                for word in doc:
                    word = word.strip()
                    if word not in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


# The rest of the functions need modifications ------------------------------
# Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    with open(filepath, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            # take out the word on each line
            word = line.strip()
            # update or add into the dictionary if word in vocab
            if word in vocab:
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
            # otherwise increment number of OOV
            else:
                if None in bow:
                    bow[None] += 1
                else:
                    bow[None] = 1
    return bow


# Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = {}
    # TODO: add your code here
    # count number of occurrence of each label (in HW2, "2016" and "2020")
    label_count = {}
    for label in label_list:
        label_count[label] = 0
    for data in training_data:
        if data['label'] in label_count:
            label_count[data['label']] += 1
    # add-1 smoothing (Laplace smoothing)
    number_of_documents = len(training_data)
    dimension = len(label_list)  # 2 dimensions in HW2
    for label in label_count:
        # probability = float(label_count[label] + smooth) / (number_of_documents + smooth * dimension)
        # use log probabilities to avoid underflow
        logprob[label] = math.log(float(label_count[label] + smooth)) - \
                         math.log(float(number_of_documents + smooth * dimension))
    return logprob


# Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    word_prob = {}
    # TODO: add your code here
    v = len(vocab)  # size of the vocabulary
    wc = 0  # the total word count with the label (regardless of being in the vocabulary or not)
    c_oov = 0  # the total count of OOVs over all documents of the given label
    for data in training_data:
        if label in data['label']:
            # count wc
            for count in data['bow'].values():
                wc += count
            # count c_oov
            if None in data['bow']:
                c_oov += data['bow'][None]
    # calculate P(word|label) for words in the given vocab
    for word in vocab:
        c = 0  # the total word count over all documents of the given label (word in vocab)
        for data in training_data:
            if label in data['label']:
                if word in data['bow']:
                    c += data['bow'][word]
        # probability = float(c + smooth) / (wc + smooth * (v + 1))
        # use log probabilities to avoid underflow
        word_prob[word] = math.log(float(c + smooth)) - math.log(float(wc + smooth * (v + 1)))
    # calculate P(None|label) for OOVs
    # probability_oov = float(c_oov + smooth) / (wc + smooth * (v + 1))
    # use log probabilities to avoid underflow
    word_prob[None] = math.log(float(c_oov + smooth)) - math.log(float(wc + smooth * (v + 1)))
    return word_prob


##################################################################################
# Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = [f for f in os.listdir(training_directory) if not f.startswith('.')]  # ignore hidden files
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    # add required entries to the returning dictionary
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, label_list)
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')
    return retval


# Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    with open(filepath, 'r') as fp:
        lines = fp.readlines()
        # start summing up final probability from ln(P(label))
        log_p_2016 = model['log prior']['2016']
        log_p_2020 = model['log prior']['2020']
        # go through each word in the file being classified
        for line in lines:
            word = line.strip()
            # summing up the log probability of the file being from 2016
            if word in model['log p(w|y=2016)']:
                log_p_2016 += model['log p(w|y=2016)'][word]
            elif None in model['log p(w|y=2016)']:
                log_p_2016 += model['log p(w|y=2016)'][None]
            # summing up the log probability of the file being from 2020
            if word in model['log p(w|y=2020)']:
                log_p_2020 += model['log p(w|y=2020)'][word]
            elif None in model['log p(w|y=2020)']:
                log_p_2020 += model['log p(w|y=2020)'][None]

        # add required entries to the returning dictionary
        retval['log p(y=2016|x)'] = log_p_2016
        retval['log p(y=2020|x)'] = log_p_2020
        # the prediction is the one with greater log probability
        # if the probabilities are equal, predict 2016 according to TA Andrew
        if log_p_2016 > log_p_2020:
            retval['predicted y'] = '2016'
        else:
            retval['predicted y'] = '2020'
    return retval


def main():
    max = 0
    best = -1
    for t in range(1, 100):
        model = train('./corpus/training/', t)
        top_level = os.listdir('./corpus/test/2016/')
        print('train = ', t)
        cnt = 0
        for d in top_level:
            if d.startswith('.'):
                # ignore hidden files
                continue
            rst = classify(model, './corpus/test/2016/' + d)
            if '2016' in rst['predicted y']:
                cnt += 1
            print(d, ': ', rst)
        print(len(top_level), 'tests from 2016, ', cnt, 'are predicted 2016')
        temp = cnt

        top_level = os.listdir('./corpus/test/2020/')
        cnt = 0
        for d in top_level:
            if d.startswith('.'):
                # ignore hidden files
                continue
            rst = classify(model, './corpus/test/2020/' + d)
            if '2020' in rst['predicted y']:
                cnt += 1
            print(d, ': ', rst)
        print(len(top_level), 'tests from 2020, ', cnt, 'are predicted 2020')
        if max < (cnt + temp):
            max = cnt + temp
            best = t
    print('best_cutoff: ', best, ' best_success: ', max)
if __name__ == "__main__":
    main()