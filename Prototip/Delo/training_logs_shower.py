
import pickle

from main import TrainingLogs

inp = input("What to add between 'training_logs_' and '.pkl' ?")

training_logs = pickle.load(open(("./saved_main/training_logs_" + inp + ".pkl"), "rb"))

print(training_logs)