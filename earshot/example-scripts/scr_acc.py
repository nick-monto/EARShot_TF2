'''
No code here yet - just trying to explain how we compute psychological (as opposed
to Keras) accuracy.

I'm not adhering specifically to Heejo's order of operations because I think we
can clean things up by just implementing the correct procedure without trying
to hew too closely to exactly how he did it.

Also, I'm slightly confused about speakers because I haven't been excluding any;
talking with Jim will clear this up. I think I'm basically correct below, however.

(Nick, this is really for you)
'''

'''
BASIC IDEA:

Every so many epochs, we do a "test" of the model.  The test involves pausing
training and running every item in the training set through the model to look at
the output.

Specifically, we compute the over-time cosine similarity to all the words in
the lexicon for each word in the training set.  So we produce data that looks
like:

    train_word, train_speaker, word_1, train, (cos(model(test_word),semantic_pattern(word_1)) over time)
    train_word, train_speaker, word_2, test, ()

The "train"/"test" identifier lets us know if we are comparing to an in-the-bag or
out-of-bag word. (We used the "test" data to look at generalization in paper 1. I
have mostly been training to all the words, so everything would be "train"). Including
the speaker just lets us conditionally average over speakers (I have yet to use this in
what I am doing).

train_word will run over all the words/speakers in the training set (might be the whole lexicon),
word_i will run over the whole lexicon. (Note - there is no associated speaker with word_i because
we don't use model output for that comparison, we use the semantic pattern.)

Ok, with all that info in hand, we can compute two things: an accuracy, and an "over-time phonological
category similarity". (NOTE 1)

We have three definitions that use these patterns to operationalize reaction time.
    1. absolute accuracy:
        this is the point in time where the cosine similarity crosses AnalyzerParameters.abs_acc_crit

    2. relative accuracy:
        this is the point in time where the activity of the most active word exceeds the next most active word
        by AnalyzerParameters.rel_acc_crit

    3. time-dependent accuracy:
        the word is "recognized" at the point when the activity of the most active word has is greater
        than the activity of the next most active word by Analyzer.td_acc_crit[1] and maintains that difference
        for Analyzer.td_acc_crit[0] time steps

NOTE: the RT is nan (inaccurate trial) if it isn't the model-input word that crosses first. For example,
suppose the word is 'cat': (NOTE 2)

    1. if the first semantic_pattern that crosses AnalyzerParameters.abs_acc_crit isn't 'cat', this is an
        error trial

    2. if that "most active word" isn't 'cat', error trial

    3. if the persistently most active word isn't 'cat', error trial

Accuracy is just 1.0 - fraction_of_words with nan RT. We get different numbers for 1,2,3 (I think we mostly use 3
right now.)

Ok, now the over-time category similarity, using the same information and the accuracy information.

FOR WORDS WHICH WERE RECOGNIZED (i.e. RT != nan):

For a given train_word, look at all the words we tested against and classify them into (overlapping) categories,
using their pronunciations (NOTE 3):

if train_word == word:
    the word we are comparing to is the target, this goes in the "average similarity to
    target" bin
    move on to the next train word

if are_cohorts(train_word,word):
    put in "average similarity to cohort" bin
    if are_neighbors(train_word,word):
        also put in "average similarity to neighbors bin" (words can be both cohors and neighbors, like "cat" and "can")
    move on the the next word

elif are_rhymes(train_word,word):
    put in "average similarity to rhymes" bin
    also put in "average similarity to neighbors bin"
    (since rhymes only differ at the initial phoneme, they are automatically also neighbors but cannot
    be cohorts)
    move on to the next word

if are_neighbors(train_word,word):
    put in "average similarity to neighbors bin"
    (you have to check neighbors here independently because you can have neighbors that
    are neither cohorts or rhymes, i.e. 'pin', and 'pan')
    move on to the next word

if you get this far, put the word in "average similarity to unrelated"
move on the the next word

You do this over all training words too - that way, for a given epoch (at which you've computed
accuracy), you have now reduced everything to five vectors of length = time (internal model time):
    1. avg. sim. to targets
    2. avg. sim. to cohorts
    3. avg. sim. to rhymes
    4. avg. sim. to neighbors
    5. avg. sim. to unrelated

NOTES:
-NOTE 1: The way things are done now, this "summary" information is produced offline - after the model is
run - basically using the all-to-all cosine overlap data. This makes some sense - you may want accuracy for
all epochs, but maybe you only want to category average for some of them. I guess if the lexicon was really
large this could be a big file, but it's probably quite compressible (so save as a pickle).

-NOTE 2: Currently, Heejo throws some of this info away - particularly, he doesn't log WHICH word the
model thought the train_word was, if the word is incorrect (you just get nan for the RT).  With more realistic
semantics, we might want to know this - like, is the "wrong" word phonologically close, semantically close,
or both to the target? Furthermore, I'm not sure we've used the actual RTs yet, but I'm sure we would like
to down the road.

-NOTE 3: This category classification should just be done once at model initialization, and maybe optionally
we should fetch a precomputed version since this only changes if the lexicon changes.
'''
