# ECSE 551 - A1
# %------------------------------------------------- Packages ----------------------------------------------------% #
import numpy               as np
import matplotlib.pyplot   as plt
# %------------------------------------------------ Data Classes -------------------------------------------------% #
class Dataset:
    def __init__(self, *arg):
        # Constructor Overloading
        # # Read from CSV file if given
        if (len(arg) == 1 and isinstance(arg[0], str)):
            self.path = arg[0]   
            self.name = arg[0].removeprefix("./").removesuffix(".csv").upper()

            # Read data from csv file
            self.data = np.genfromtxt(arg[0], delimiter=',')

        # # Assign array if it is np.array
        elif (len(arg) == 2 and isinstance(arg[0], np.ndarray)):
            self.path = None
            self.name = arg[1]
            self.data = np.copy(arg[0])
        
        else:
            raise TypeError(f'Expected arguments to be either:\nA)Path to a CSV file\nB)Numpy Array + Name')

        # Data Dim
        (self.rows, self.cols) = self.data.shape

        # Extract the classes from the data
        self.Class1 = Dataset.Classifier(self.data[self.data[:, -1] == 1])
        self.Class0 = Dataset.Classifier(self.data[self.data[:, -1] == 0])

    # Purpose: Compare the mean of each feature for both classes
    # # Plots the mean of each Feature
    def plotDatasetFeatureMeans(self):     
        plt.figure(figsize=(15,8))
        self.Class1.plotFeatureMean()
        self.Class0.plotFeatureMean()
        plt.title(f'{self.name}\nFeature Means')
    
    # Purpose: Compare the mean of each feature for both classes
    # # Plots the mean of each Feature
    def plotDatasetFeatureStd(self):     
        plt.figure(figsize=(15,8))
        self.Class1.plotFeatureStd()
        self.Class0.plotFeatureStd()
        plt.title(f'{self.name}\nFeature Standard Deviation')

    # Purpose: Compare the spread of features for both classes
    # # Plots the sorted version of each Feature
    def plotDatasetFeatureSpread(self):
        x1 = range(0, self.Class1.rows)
        x0 = range(0, self.Class0.rows)

        for f in range(0, self.cols-1):
            plt.figure(figsize=(15,8))
            plt.plot(x1, np.sort(self.Class1.data[:, f]),
                     label='Class 1')
            plt.plot(x0, np.sort(self.Class0.data[:, f]),
                     label='Class 0')
            plt.legend(); plt.grid(linestyle = '--')
            plt.xlabel('Observation'); plt.ylabel('Value')
            plt.title(f'{self.name}\n Feature {f+1}') 
    
    # TODO: Add histogram plot

    def plot_histogram(self):

        plt.style.use('classic')
        plt.rcParams.update({
                         "lines.linewidth" : 3,                          # you should make this general for all plots
                         "font.size" : 17,
                         "figure.constrained_layout.use" : True,
                         "hist.bins" : 20 
                         })

        fig, ax = plt.subplots(int(np.ceil(self.Class1.X.shape[1]/4)), 4, sharex=False)

        data = np.concatenate((self.Class1.X, self.Class0.X), axis=0)

        for i in range(data.shape[1]):
            ax[int(i/4), i%4].hist([self.Class1.X[:, i], self.Class0.X[:, i]], density=True, label=['class1', 'class0'], color=['blue', 'red'])
            ax[int(i/4), i%4].set_title('Feature ' + str(i+1))

            #show average and standard deviation
            textstr = '\n'.join((
                r'$\mu_{class0}=%.2f$' % (np.mean(self.Class0.X[:, i]), ),
                r'$\sigma_{class0}=%.2f$' % (np.std(self.Class0.X[:, i]), ), 
                r'$\mu_{class1}=%.2f$' % (np.mean(self.Class1.X[:, i]), ),
                r'$\sigma_{class1}=%.2f$' % (np.std(self.Class1.X[:, i]), )))

            ax[int(i/4), i%4].text(0.67, 0.60, textstr, transform=ax[int(i/4), i%4].transAxes)
            
        # ax[int(i/4), i%4].text(0.5, 0.5, 'mean: ' + str(np.round(np.mean(data[:, i]), 2)) + 'standard deviation: ' + str(np.round(np.std(data[:, i]), 2)), horizontalalignment='center', verticalalignment='center')
        
        ax[0, 0].legend(loc='lower right')

    # TODO: Add subplots
    
    # Purpose: Run tests
    @staticmethod
    def analyzeData(dataSet, Dstd=False, Dmean=False, Dspread=False, Hist=False):
        # Compare the spread and mean of the features of each class
        if(Dstd):    dataSet.plotDatasetFeatureStd()
        if(Dmean):   dataSet.plotDatasetFeatureMeans()
        if(Dspread): dataSet.plotDatasetFeatureSpread() 
        if(Hist): dataSet.plot_histogram()
        plt.show()    
        
    class Classifier:
        def __init__(self, data):
            self.data = data
            self.X, self.Y = data[:,:-1] , data[:,-1]
            self.name = "Class " + str(data[0,-1])

            # Data Dim
            (self.rows, self.cols) = data.shape
            self.featuresList      = range(1, self.cols-1)

            # Data stats
            self.colAvrg = np.mean(data[:,:-1], axis=0)
            self.colStd  = np.std(data[:,:-1], axis=0)

        def plotFeatureMean(self):
            marker = 'o' if self.data[0,-1]==1 else 'x'
            plt.plot(self.featuresList, self.colAvrg[:-1], 
                    marker,  label=self.name, markersize=10)
            plt.legend(); plt.grid(linestyle = '--')
            plt.xticks(np.arange(1, self.cols, 1.0))
            plt.xlabel('Features'); plt.ylabel('Mean Value')
            plt.title(self.name)
            
        def plotFeatureStd(self):
            marker = 'o' if self.data[0,-1]==1 else 'x'
            plt.plot(self.featuresList, self.colStd[:-1], 
                    marker,  label=self.name, markersize=10)
            plt.legend(); plt.grid(linestyle = '--')
            plt.xticks(np.arange(1, self.cols, 1.0))
            plt.xlabel('Features'); plt.ylabel(r'$\sigma$ Value')
            plt.title(self.name)