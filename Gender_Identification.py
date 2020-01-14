import nltk
from nltk.corpus import names
import random

def gender_features(word):
	feature = word[-1:]
	return {'first_and_last_two_letters': feature}
# gender_features('Shrek') = {'last_letter': 'k'}

male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
labeled_names = male_names + female_names
random.shuffle(labeled_names)
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
#entries are    ({'last_letter': 'g'}, 'male')
train_set, test_set = featuresets[700:], featuresets[:300]

classifier = nltk.NaiveBayesClassifier.train(train_set)

ans1 = classifier.classify(gender_features('Keanu'))
ans2 = classifier.classify(gender_features('Kate'))

print("Keanu is:", ans1)
print("Kate is:", ans2)

classifier.show_most_informative_features(5)
print(nltk.classify.accuracy(classifier, test_set))





