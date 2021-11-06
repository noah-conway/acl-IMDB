import utilities as ut

BASE_DIR = "/home/noah/PycharmProjects/aclimdb/aclImdb"
TESTING_DIR = "{}/test".format(BASE_DIR)
TRAINING_DIR = "{}/train".format(BASE_DIR)

#parameters
MIN_COUNT = 2
MAX_FILES = 10


vocab = ut.generate_vocab(TESTING_DIR, MIN_COUNT, MAX_FILES)
features, labels = ut.load_data(TESTING_DIR, vocab, MAX_FILES)


print(labels)












