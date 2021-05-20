import joblib
from sklearn.model_selection import train_test_split as splitter
from sklearn.neural_network import MLPClassifier

print("Loading data..................................")
with open("classify.csv", mode="r") as f:
    old_raw_lines = f.readlines()
print("Encoding data.................................")
raw_lines = []
for i in range(len(old_raw_lines)):
    if i % 10 == 0:
        raw_lines.append(old_raw_lines[i])
data = []
target = []
for line in raw_lines:
    str_temp = line.strip("\n").split(",")
    target.append(str_temp[-1])
    temp = []
    for item in str_temp[0:-1]:
        temp.append(int(item)/255)
    data.append(temp)
print("Training data totally have " + str(len(data)) + " pieces.")
label_dict = {}
for item in target:
    if not label_dict.keys().__contains__(item):
        label_dict[item] = 0
    label_dict[item] += 1
print("Training labels totally have " + str(len(label_dict.keys())) + " classes.")
print("Training model..........................")
X_train, X_test, Y_train, Y_test = splitter(data, target, test_size=0.2, random_state=30)
model = MLPClassifier([256, 256], learning_rate_init=0.001, activation='relu',
                      solver='adam', alpha=0.0001, max_iter=10000)  # 神经网络
print('start training..........................')
model.fit(data, target)
print('end training............................')
joblib.dump(model, "SimpleClassifier.pkl")
train_accuracy = model.score(X_train, Y_train)
test_accuracy = model.score(X_test, Y_test)
print("Classifier has %.4f percent accuracy on train set." % train_accuracy*100)
print("Classifier has %.4f percent accuracy on test set." % test_accuracy*100)

