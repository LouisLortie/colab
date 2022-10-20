#%% #@title Import list

# from google.colab import drive                    # Need to activate this if using Google Colab (with drive)
import csv
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import ecse_551_assignment1


#@ Function that counts the number of features
def feature_count(data):
  return data.shape[1] - 1         # The last column is the class label

# Function that separates the input and output data
def get_xy_data(data):
    x = data[:,:-1]
    y = data[:,-1].reshape(-1,1)
    return x, y


#%% #@title Setting rcParams
plt.style.use('classic')
plt.rcParams.update({
                        #  "text.usetex": True,
                        #  "font.family": "serif",
                        #  "font.sans-serif": ["Times"],
                         "lines.linewidth" : 3,
                         "font.size" : 12,
                         "figure.constrained_layout.use" : True,
                         "hist.bins" : 20 
                         })


#%%    #@ Function that classifies the training data into the two binary classes
def data_separation(data):

  count_class1 = 0
  count_class0 = 0
  
  for i in range(data.shape[0]) :      # all rows of data

    if data[i, -1] == 0 :
      if count_class0== 0 :
        data_class0 = data[[i], :]
      else :
        data_class0 = np.concatenate((data_class0, data[[i], :]), axis=0)
      count_class0+= 1

    elif data[i, -1] == 1 :
      if count_class1 == 0 :
        data_class1 = data[[i], :]
      else :
        data_class1 = np.concatenate((data_class1, data[[i], :]), axis=0)
      count_class1 += 1

  return data_class1, data_class0


#%% #@ Machine learning method: Classicifiation Discriminative learning
class DiscriminativeLearning():

    def __init__(self, data, epsilon):
        self.data = data
        np.random.shuffle(self.data)
        self.x, self.y = get_xy_data(self.data)
        self.x = np.insert(self.x, 0, 1, axis=1)
        self.epsilon = epsilon
        self.w0 = np.zeros((self.x.shape[1], 1))

    
    def logistic_function(self, w, xi):
        a = w.T @ xi
        s = 1 / (1 + np.exp(-a))
        return s


    def fit(self):

        w_old = self.w0
        w_new = w_old

        while True:

            delta = 0
            w_old = w_new

            for i in range(self.x.shape[0]):

                step_size = 1 / (1 + i)
                xi = self.x[i].reshape(-1,1)
                yi = self.y[i]
                delta = delta - xi @ (yi - self.logistic_function(w_old, xi))

            w_new = w_old - step_size * delta
            error = np.linalg.norm(w_new - w_old, ord =2)

            if error < self.epsilon:
                
                break
        
        self.w = w_new

        return 0


    def predict(self, x):

        x = x.reshape(-1,1)
        x = np.insert(x, 0, 1, axis=0)
        p = self.w.T @ x
        
        if p > 0.5:
            y = 1
        elif p == 0.5:
            # randomly choose 0 or 1
            y = np.random.randint(0,2)
        else:
            y = 0
        return y



#%% #@title GenerativeLearning for LDA class: Generative model for binary classification

class GenerativeLearning_lda:

  def __init__(self, data):
    self.data = data
    self.data_class1, self.data_class0 = data_separation(data)
    self.data = np.concatenate((self.data_class1, self.data_class0), axis=0)
    self.feat_count = feature_count(self.data)
    self.class1_count = self.data_class1.shape[0]
    self.class0_count = self.data_class0.shape[0]
    self.class1_prior = self.class1_count / self.data.shape[0]
    self.class0_prior = self.class0_count / self.data.shape[0]
    self.class1_x = self.data_class1[:,:-1]
    self.class0_x = self.data_class0[:,:-1]
    self.cov = np.zeros((self.data.shape[0], self.data.shape[0]))


  def fit_lda_linear_plt(self):                # used for the special plot
    
    self.w0 = np.log(self.class1_prior) - np.log(self.class0_prior) - 0.5 * self.class1_mean @ np.linalg.inv(self.cov) @ self.class1_mean.T + 0.5 * self.class0_mean @ np.linalg.inv(self.cov) @ self.class0_mean.T
    self.w1 = np.linalg.inv(self.cov) @ (self.class1_mean - self.class0_mean).T
    self.w = np.concatenate((self.w0, self.w1))

    return self.w
    

  def fit(self):

    self.class1_mean = np.mean(self.data_class1[:,:-1], axis=0).reshape(1, -1)
    self.class0_mean = np.mean(self.data_class0[:,:-1], axis=0).reshape(1, -1)

    self.cov = ((self.class1_x - self.class1_mean).T @ (self.class1_x - self.class1_mean) + (self.class0_x - self.class0_mean).T @ (self.class0_x - self.class0_mean)) / (self.data.shape[0] - 2)


  def predict(self, data):

    delta1 = data @ np.linalg.inv(self.cov) @ self.class1_mean.T - 0.5 * self.class1_mean @ np.linalg.inv(self.cov) @ self.class1_mean.T + np.log(self.class1_prior)

    delta0 = data @ np.linalg.inv(self.cov) @ self.class0_mean.T - 0.5 * self.class0_mean @ np.linalg.inv(self.cov) @ self.class0_mean.T + np.log(self.class0_prior)

    if delta1 > delta0:

      return 1

    elif delta1 == delta0:

        return np.random.randint(0,2)

    else:

      return 0


  def predict_plt(self, x):

    self.y = x @ self.w[1:] + self.w[0]
    # print(self.y)
    
    return self.y


#%% #@title GenerativeLearning for QDA class: Generative model for binary classification

class GenerativeLearning_qda:

  def __init__(self, data):
    self.data = data
    self.data_class1, self.data_class0 = data_separation(data)
    self.data = np.concatenate((self.data_class1, self.data_class0), axis=0)
    self.feat_count = feature_count(self.data)
    self.class1_count = self.data_class1.shape[0]
    self.class0_count = self.data_class0.shape[0]
    self.class1_prior = self.class1_count / self.data.shape[0]
    self.class0_prior = self.class0_count / self.data.shape[0]
    self.class1_x = self.data_class1[:,:-1]
    self.class0_x = self.data_class0[:,:-1]
    self.cov = np.zeros((self.data.shape[0], self.data.shape[0]))


  def fit(self):

    self.class1_mean = np.mean(self.data_class1[:,:-1], axis=0).reshape(1, -1)
    self.class0_mean = np.mean(self.data_class0[:,:-1], axis=0).reshape(1, -1)

    self.cov_class1 = (self.class1_x - self.class1_mean).T @ (self.class1_x - self.class1_mean) / (self.class1_count - 1)
    self.cov_class0 = (self.class0_x - self.class0_mean).T @ (self.class0_x - self.class0_mean) / (self.class0_count - 1)
    

  def predict(self, data):

    delta1 = -0.5*np.log(np.linalg.det(self.cov_class1)) - 0.5*(data-self.class1_mean) @ np.linalg.inv(self.cov_class1) @ (data-self.class1_mean).T + np.log(self.class1_prior)

    delta0 = -0.5*np.log(np.linalg.det(self.cov_class0)) - 0.5*(data-self.class0_mean) @ np.linalg.inv(self.cov_class0) @ (data-self.class0_mean).T + np.log(self.class0_prior)


    if delta1 > delta0:

      return 1

    elif delta1 == delta0:

        np.random.randint(0,2)

    else:

      return 0


#%% #@title Machine learning method: K Fold Validation
class KFoldValidation():
    def __init__(self, data, model, k):
        self.data = data
        np.random.shuffle(self.data)
        self.k = k
        self.n = self.data.shape[0]
        self.indices = np.arange(self.n)
        self.fold_size = int(self.n / self.k)
        self.fold_indices = np.array_split(self.indices, self.k)
        self.model = model

    def train_valid_split(self, i):

        if i >= self.k:
            raise ValueError("The fold number is out of range")

        valid_indices = self.fold_indices[i]
        train_indices = np.delete(self.indices, valid_indices) 
        train_data = self.data[train_indices]
        valid_data = self.data[valid_indices]

        return train_data, valid_data

    def kfold_validation(self):

        count = np.array([0, 0, 0, 0]).reshape(-1,)                             # This array is used to count the number of true positive, true negative, false positive and false negative.

        for i in range(self.k):

            error = 0
            train_data, valid_data = self.train_valid_split(i)
            valid_x, valid_y = get_xy_data(valid_data)
            
            if self.model == "disc_l":
                model = DiscriminativeLearning(train_data, 0.001)
            elif self.model == "lda":
                model = GenerativeLearning_lda(train_data)
            elif self.model == "qda":
                model = GenerativeLearning_qda(train_data)

            model.fit()

            for j in range(valid_x.shape[0]):                      # This should be done in a matrix way. Let the input to predict be a matrix.

                xi = valid_x[j]
                yi = valid_y[j]
                y_pred = model.predict(xi)

                count = error_measures(y_pred, yi, count)          # This function is used to count the number of true positive, true negative, false positive and false negative.

        return count/self.k


def error_measures(y_pred, y_val, count):

    if y_pred == 1 and y_val == 1:
        count[0] += 1
    elif y_pred == 0 and y_val == 0:
        count[1] += 1
    elif y_pred == 1 and y_val == 0:
        count[2] += 1
    elif y_pred == 0 and y_val == 1:
        count[3] += 1

    return count


class accu_eval():
    def __init__(self, count):
        self.count = count

    def common_measures(self):
        accuracy = (self.count[0] + self.count[1]) / (self.count[0] + self.count[1] + self.count[2] + self.count[3])
        precision = self.count[0] / (self.count[0] + self.count[2])
        recall = self.count[0] / (self.count[0] + self.count[3])
        sensitivity = self.count[0] / (self.count[0] + self.count[3])
        specificity = self.count[1] / (self.count[1] + self.count[2])
        false_positive_rate = self.count[2] / (self.count[2] + self.count[1])
        true_positive_rate = recall
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score                     # change if you want to see other measures


    def mcc(self):                          # This function is used to calculate the Matthews correlation coefficient.

        mcc = (self.count[0] * self.count[1] - self.count[2] * self.count[3]) / np.sqrt((self.count[0] + self.count[2]) * (self.count[0] + self.count[3]) * (self.count[1] + self.count[2]) * (self.count[1] + self.count[3]))

        return mcc


#%% #@ test_loop function: Testing loop that tries several data samples and returns some sort of accuracy

def test_loop(data, model, model_type):

    count_class1 = 0
    count_class0 = 0
    count_false1 = 0
    count_false0 = 0
    test = 0

    for i in range(data.shape[0]):

        if model_type == 'lda':
            prediction = model.predict(data[[i], :-1])
        elif model_type == "qda":
            prediction = model.predict(data[[i], :-1])
        elif model_type == "disc":
            prediction = model.predict(data[[i], :-1])

        if prediction == 1:

            if data[i, -1] == 0:

                count_false1 += 1

            count_class1 += 1

        elif prediction == 0:

            if data[i, -1] == 1:

                count_false0 += 1

            count_class0 += 1

    print('number of class1 samples: ', count_class1)
    print('number of class0 samples: ', count_class0)
    print('number of false class1 samples: ', count_false1)
    print('number of false class0 samples: ', count_false0)
    print('accracy: ', 1- (count_false1 + count_false0) / (count_class1 + count_class0))


#%% #@title augment_ones function: Function that augments the data with a column of ones

def augment_ones(data):
    ones = np.ones((data.shape[0], 1))
    return np.concatenate((data, ones), axis=1)


#%% #@title increase_complexity function: Function that increases the complexity of the data

def increase_complexity(data, feat_num):
    # data = augment_ones(data)                                    # could be used to augment the data with a column of ones

    last_col = data[:, [-1]]
    data = data[:, :-1]

    for i in range(feat_num.shape[0]):
        data = np.concatenate((data, data[:, [feat_num[i]]]**2), axis=1)   # augment the matrix by adding a column of the square of the feature

    data = np.concatenate((data, last_col), axis=1)       # add the last column back to the matrix

    return 
    

#%% #@title fold_prep: Separates data into testing, validation and testing sets

def fold_prep(data, train_num, validation_num, test_num):
    # np.random.shuffle(data)
    data_train = data[:train_num, :]
    data_val = data[train_num:train_num+validation_num, :]
    data_test = data[train_num+validation_num:train_num+validation_num+test_num, :]
    return data_train, data_val, data_test 


#%% #@title plot hist: Function that plots the histogram of one feature
# The number of bins can be changed via rcParams above.

def plot_hist(data_class1, data_class0):
    fig, ax = plt.subplots(int(np.ceil(data_class1.shape[1]/4)), 4, sharex=False)

    data = np.concatenate((data_class1, data_class0), axis=0)

    for i in range(data.shape[1]):
        ax[int(i/4), i%4].hist([data_class1[:, i], data_class0[:, i]], density=True, label=['class1', 'class0'], color=['blue', 'red'])
        ax[int(i/4), i%4].set_title('Feature ' + str(i+1))

        #show average and standard deviation
        textstr = '\n'.join((
            r'$\mu_{class0}=%.2f$' % (np.mean(data_class0[:, i]), ),
            r'$\sigma_{class0}=%.2f$' % (np.std(data_class0[:, i]), ), 
            r'$\mu_{class1}=%.2f$' % (np.mean(data_class1[:, i]), ),
            r'$\sigma_{class1}=%.2f$' % (np.std(data_class1[:, i]), )))

        ax[int(i/4), i%4].text(0.70, 0.70, textstr, transform=ax[int(i/4), i%4].transAxes)
            
        # ax[int(i/4), i%4].text(0.5, 0.5, 'mean: ' + str(np.round(np.mean(data[:, i]), 2)) + 'standard deviation: ' + str(np.round(np.std(data[:, i]), 2)), horizontalalignment='center', verticalalignment='center')
        
    ax[0, 0].legend(loc='center right')


#   ax[0,0].hist(data_class1, color="b", density=False, label=r"Class y = 1")
#   ax[1,0].hist(data_class0, color="r", density=False, label=r"Class y = 0")
#   ax[0,0].set_title("Feature %i distribution comparison between classes" %(feat_num + 1))
#   ax[0,0].legend(loc='upper right')
#   ax[1,0].legend(loc='upper right')
    plt.tight_layout

    return 0


#%% #@title prediction_line: Function that plots the regression line

# Only used to show linear LDA. Not used for QDA.

def prediction_line(x_class1, x_class0, w): 

    count_false1 = 0
    count_false0 = 0
    
    x = (np.concatenate((x_class1, x_class0), axis=0)).reshape(-1, 1)
    
    fig, ax = plt.subplots(1, 1)
    y_class1 = x_class1 @ w[1:] + w[0]
    y_class0 = x_class0 @ w[1:] + w[0]
    range1 = np.linspace(np.min(x), np.max(x), y_class1.shape[0])
    range2 = np.linspace(np.min(x), np.max(x), y_class0.shape[0])

    for i in range(y_class1.shape[0]):
        if y_class1[i] < 0:
            count_false0 += 1

    for i in range(y_class0.shape[0]):
        if y_class0[i] > 0:
            count_false1 += 1

    print("Number of false positives: %i" %(count_false1))
    print("Number of false negatives: %i" %(count_false0))
    print("Total of class1: %i" %(y_class1.shape[0]))
    print("Total of class0: %i" %(y_class0.shape[0]))

    ax.scatter(range1, y_class1, color="b", label=r"Class y = 1")
    ax.scatter(range2, y_class0, color="r", label=r"Class y = 0")
    ax.plot([np.min(x), np.max(x)], [0, 0], color="g")
    ax.set_title("Predictions")
    ax.legend(loc='upper right')
    plt.tight_layout
    
    return 0


    #%% #@title Main function
def main():

    # drive.mount('/content/drive')                    # Activate on drive
    # Reading air quality data
    # aq_csv = pd.read_csv("/content/drive/MyDrive/ecse_551/assignment1/air_quality.csv")                 # Active on drive
    aq_csv = pd.read_csv("/home/louis/Documents/mcgill_classes/ecse_551/assignment1/colab/air_quality.csv", header=None)   # Active on colab

    # Reading liver patient data
    # lp_csv = pd.read_csv("/content/drive/MyDrive/ecse_551/assignment1/liver_patient.csv")               # Active on drive
    lp_csv = pd.read_csv("/home/louis/Documents/mcgill_classes/ecse_551/assignment1/colab/liver_patient.csv", header=None)   # Active on colab
    

    # Array
    aq_data = np.array(aq_csv)
    lp_data = np.array(lp_csv)

    # Divide into inputs and outputs
    aq_x, aq_y = get_xy_data(aq_data)
    lp_x, lp_y = get_xy_data(lp_data)


    feat = 7   # Feature to analyse.

    # aq_data_aug = increase_complexity(aq_data, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    data_class1, data_class0 = data_separation(lp_data)

    # plot_hist(data_class1[:, feat], data_class0[:, feat], feat)
    plot_hist(data_class1[:, :-1], data_class0[:, :-1])

    # All model creation happens here

    model_lda = GenerativeLearning_lda(aq_data)
    model_qda = GenerativeLearning_qda(aq_data)
    discriminative_learning = DiscriminativeLearning(lp_data, 0.001) # shuffles data for now

    model_lda.fit()
    model_qda.fit()
    w = model_lda.fit_lda_linear_plt()

    model_label = "lda"                 # can be changed to "lda", "qda" or "disc"

    # prediction_line(data_class1[:, :-1], data_class0[:, :-1], w)

    test_loop(aq_data, model_lda, model_label)

    # to unshuffle data just do. does the work easier than unshuffling data using a dedicated function.
    aq_data = np.array(aq_csv)
    lp_data = np.array(lp_csv)

    k_fold_validation = KFoldValidation(aq_data, model_label, 2)         # "disc_l" or "lda" or "qda"
    count = k_fold_validation.kfold_validation()
    print("model error is: ", count)

    error_metric = accu_eval(count)

    f1_score = error_metric.common_measures()
    print("f1 score is: ", f1_score)

    mcc = error_metric.mcc()
    print("mcc is: ", mcc)

    plt.show()
  
if __name__ == '__main__':
    main()
