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
                         "hist.bins" : 100 
                         })


#%% #@ Machine learning method: Classicifiation Discriminative learning
class DiscriminativeLearning():

    def __init__(self, x, y, epsilon):
        self.x = np.insert(x, 0, 1, axis=1)
        self.y = y
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
        return w_new

    def predict(self, w, x):
        x = x.reshape(-1,1)
        x = np.insert(x, 0, 1, axis=0)
        p = w.T @ x
        if p > 0.5:
            y = 1
        elif p == 0.5:
            # randomly choose 0 or 1
            y = np.random.randint(0,2)
        else:
            y = 0
        return y


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
    self.class1_mean = np.mean(self.data_class1[:,:-1], axis=0).reshape(1, -1)
    self.class0_mean = np.mean(self.data_class0[:,:-1], axis=0).reshape(1, -1)
    self.class1_x = self.data_class1[:,:-1]
    self.class0_x = self.data_class0[:,:-1]
    self.cov = np.zeros((self.data.shape[0], self.data.shape[0]))

    # Notice the difference in the covariance matrix calculation

    self.cov_class1 = (self.class1_x - self.class1_mean).T @ (self.class1_x - self.class1_mean) / (self.class1_count - 1)
    self.cov_class0 = (self.class0_x - self.class0_mean).T @ (self.class0_x - self.class0_mean) / (self.class0_count - 1)

    self.cov = ((self.class1_x - self.class1_mean).T @ (self.class1_x - self.class1_mean) + (self.class0_x - self.class0_mean).T @ (self.class0_x - self.class0_mean)) / (self.data.shape[0] - 2)

    # need to check better way to find inverse of covariance matrix


  def fit_lda_linear_plt(self):
    
    self.w0 = np.log(self.class1_prior) - np.log(self.class0_prior) - 0.5 * self.class1_mean @ np.linalg.inv(self.cov) @ self.class1_mean.T + 0.5 * self.class0_mean @ np.linalg.inv(self.cov) @ self.class0_mean.T
    self.w1 = np.linalg.inv(self.cov) @ (self.class1_mean - self.class0_mean).T
    self.w = np.concatenate((self.w0, self.w1))

    return self.w
    

  def fit_lda(self, data, label):


    if label == 1:

        delta = data @ np.linalg.inv(self.cov) @ self.class1_mean.T - 0.5 * self.class1_mean @ np.linalg.inv(self.cov) @ self.class1_mean.T + np.log(self.class1_prior)

    elif label == 0:

        delta = data @ np.linalg.inv(self.cov) @ self.class0_mean.T - 0.5 * self.class0_mean @ np.linalg.inv(self.cov) @ self.class0_mean.T + np.log(self.class0_prior)

    return delta


  def predict_lda(self, data):

    if self.fit_lda(data, 1) > self.fit_lda(data, 0):

      return 1

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
    self.class1_mean = np.mean(self.data_class1[:,:-1], axis=0).reshape(1, -1)
    self.class0_mean = np.mean(self.data_class0[:,:-1], axis=0).reshape(1, -1)
    self.class1_x = self.data_class1[:,:-1]
    self.class0_x = self.data_class0[:,:-1]
    self.cov = np.zeros((self.data.shape[0], self.data.shape[0]))

    # Notice the difference in the covariance matrix calculation

    self.cov_class1 = (self.class1_x - self.class1_mean).T @ (self.class1_x - self.class1_mean) / (self.class1_count - 1)
    self.cov_class0 = (self.class0_x - self.class0_mean).T @ (self.class0_x - self.class0_mean) / (self.class0_count - 1)

    self.cov = ((self.class1_x - self.class1_mean).T @ (self.class1_x - self.class1_mean) + (self.class0_x - self.class0_mean).T @ (self.class0_x - self.class0_mean)) / (self.data.shape[0] - 2)

    # need to check better way to find inverse of covariance matrix


  def fit_qda(self, data, label):      # Should be fine, but has not been tested out


      if label == 1:

          delta = -0.5*np.log(np.linalg.det(self.cov_class1)) - 0.5*(data-self.class1_mean) @ np.linalg.inv(self.cov_class1) @ (data-self.class1_mean).T + np.log(self.class1_prior)

      elif label == 0:

          delta = -0.5*np.log(np.linalg.det(self.cov_class0)) - 0.5*(data-self.class0_mean) @ np.linalg.inv(self.cov_class0) @ (data-self.class0_mean).T + np.log(self.class0_prior)
  
      return delta


  def predict_qda(self, data):

    if self.fit_qda(data, 1) > self.fit_qda(data, 0):

      return 1

    else:

      return 0


#%% #@title Machine learning method: K Fold Validation
class KFoldValidation():
    def __init__(self, data, model, k):
        self.data = data
        self.k = k
        self.n = self.data.shape[0]
        self.indices = np.arange(self.n)
        self.fold_size = int(self.n / self.k)
        self.fold_indices = np.array_split(self.indices, self.k)
        self.model = model

    def train_test_split(self, i):
        test_indices = self.fold_indices[i]
        train_indices = np.delete(self.indices, test_indices)
        train_data = self.data[train_indices]
        test_data = self.data[test_indices]
        return train_data, test_data

    def kfold_validation(self):
        error_val = []
        for i in range(self.k):
            error = 0
            train_data, test_data = self.train_test_split(i)
            train_x, train_y = get_xy_data(train_data)
            test_x, test_y = get_xy_data(test_data)
            
            if self.model == 1:
                model = DiscriminativeLearning(train_x, train_y, 0.001)
            # elif self.model == 2:
            #     # TODO: Add model 2
            # elif self.model == 3:
            #     # TODO : Add model 3
            w = model.fit()
            for j in range(test_x.shape[0]):
                xi = test_x[j]
                yi = test_y[j]
                y_pred = model.predict(w, xi)
                error = error + (yi - y_pred)**2
            error_val.append(error)
        error_val = np.average(error_val)
        return error_val


#%% #@ test_loop function: Testing loop that tries several data samples and returns some sort of accuracy

def test_loop(data, model, model_type):

    count_class1 = 0
    count_class0 = 0
    count_false1 = 0
    count_false0 = 0
    test = 0

    if model_type == 'lda':

        for i in range(data.shape[0]):

            prediction = model.predict_lda(data[[i], :-1])

            if prediction == 1:

                if data[i, -1] == 0:

                    count_false1 += 1

                count_class1 += 1

            elif prediction == 0:

                if data[i, -1] == 1:

                    count_false0 += 1

                count_class0 += 1

    elif model_type == "qda":

        for i in range(data.shape[0]):

            prediction = model.predict_qda(data[[i], :-1])

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

def plot_hist(feat_class1, feat_class0, feat_num):
  fig, ax = plt.subplots(2, 1, sharex=True)
  ax[0].hist(feat_class1, color="b", label=r"Class y = 1")
  ax[1].hist(feat_class0, color="r", label=r"Class y = 0")
  ax[0].set_title("Feature %i distribution comparison between classes" %(feat_num + 1))
  ax[0].legend(loc='upper right')
  ax[1].legend(loc='upper right')
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
    lp_csv = pd.read_csv("/home/louis/Documents/mcgill_classes/ecse_551/assignment1/colab/air_quality.csv", header=None)   # Active on colab
    

    # Array
    aq_data = np.array(aq_csv)
    lp_data = np.array(lp_csv)

    # Divide into inputs and outputs
    aq_x, aq_y = get_xy_data(aq_data)
    lp_x, lp_y = get_xy_data(lp_data)


    feat = 2   # Feature to analyse.

    # aq_data_aug = increase_complexity(aq_data, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    data_class1, data_class0 = data_separation(aq_data)

    plot_hist(data_class1[:, feat], data_class0[:, feat], feat)

    model_lda = GenerativeLearning_lda(aq_data)
    model_qda = GenerativeLearning_qda(aq_data)

    w = model_lda.fit_lda_linear_plt()
    prediction_line(data_class1[:, :-1], data_class0[:, :-1], w)

    test_loop(aq_data, model_qda, "qda")


    plt.show()

    # k_fold_validation = KFoldValidation(lp_data, 1, 2)
    # model_error = k_fold_validation.kfold_validation()
    # print(model_error)
  
if __name__ == '__main__':
    main()
# %%
