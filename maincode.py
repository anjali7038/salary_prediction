import os                                        #used to get file from current directory
import numpy as np                               #used to storing data in array format and manipulation
import pandas as pd                                        #dataframe creates dataset from file
import matplotlib.pyplot as plt                            #graph plotting
from sklearn.model_selection import train_test_split        # used to split training and test data
from sklearn.metrics import r2_score       #to predict model performance
from sklearn.linear_model import LinearRegression


def welcome():            #welcome fn to print msg
    print("Welcome to salary prediction system")
    print("press enter key to start")
def checkcsv():            #get the csv file from the directory
    csv_files=[]
    curr_dir=os.getcwd()                 #to get th current directory
    content_list=os.listdir(curr_dir)    #to fetch files from current directory
    for x in content_list:
        if x.split(".")[-1]=='csv':
            csv_files.append(x)
    if(len(csv_files)==0):
        return("No csv file in directory")
    else:
        return(csv_files)
def display_and_select(csv_files):
    i=0
    for file in csv_files:
        print(i,file,sep=" ")
        i+=1
    return(csv_files[int(input("select file"))])    #the corresponding input index file is returned for dataset
def graph(x_train,y_train,regressionObject,x_test,y_test,y_pred):
    plt.scatter(x_test,y_pred,color='black',label='predicted test data')
    plt.scatter(x_train,y_train,color='red',label='training data')       #to scatter plot bw x and y
    plt.plot(x_train,regressionObject.predict(x_train),color="blue",label="best fit") #draw regression line
    plt.scatter(x_test,y_test,color="green",label="test data")
    plt.title("salary vs experience")        #to give title to graph
    plt.xlabel("years of experience")
    plt.ylabel("salary")
    plt.legend()    #to show label
    plt.show()     #to show graph
def main():
    welcome()
    try:
        csv_files=checkcsv()
        if(csv_files=="No csv file in directory"):
            raise FileNotFoundError("No csv in directory")
        csv_file=display_and_select(csv_files)
        print("csv file is selected")
        print("reading csv file...")
        print("creating dataset")
        dataset=pd.read_csv(csv_file)    #to read the csv file and create dataset
        x=dataset.iloc[:,:-1].values      #to get values from last column
        y=dataset.iloc[:,-1].values       #to get values except for last col from csv file
        s=float(input("enter test data size(between 0 and 1)"))
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=s) #fn to split into training and testing dataset
        print("model creation in progress...")
        regressionObject= LinearRegression()        #create object of linear regression
        regressionObject.fit(x_train,y_train)
        print("press enter to predict training data in training model")
        input()
        y_pred=regressionObject.predict(x_test)         #fn to predict test data
        i=0
        print("x_test","   ","y_test","   ","y_pred")
        while(i<len(x_test)):
            print(x_test[i],'   ',y_test[i],'   ',y_pred[i])
            i+=1
        print("press enter to see graphical view")
        input()
        graph(x_train,y_train,regressionObject,x_test,y_test,y_pred) #prints graph
        r2=r2_score(y_test,y_pred)
        print("our model is %2.2f%% accurate" %(r2*100))     #accuracy of model
        print("enter the experience in years to predict salary (comma seperated)")
        exp=[float(e) for e in input().split(',')]    #taking input experience as list
        ex=[]
        for x in exp:
            ex.append([x])
        experience=np.array(ex)
        salaries=regressionObject.predict(experience)
        plt.scatter(experience,salaries,color='black')
        plt.xlabel("years of experience")
        plt.ylabel("predicted salaries")
        plt.show() 
        d=pd.DataFrame({'Experience':exp,'salaries':salaries})   #dataframe to store experience and predicted salary
        print(d)        
    except FileNotFoundError:
     print("No csv file found")  
    print("press enter to exit")
    input()  


if __name__=="__main__":   #to start the program from this fn
    main()


