import pandas as pan;
import sklearn as sk;
import numpy as num;
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import functools
import seaborn as sns

input_filename = 'PlayerInputInformation_multiple.csv';
filename = 'SoccerPlayerStatistics.csv'
attributes_to_be_removed = ['Name', 'Preffered_Position', 'Contract_Expiry'];
categorical_attributes = ['Club_Kit', 'Nationality', 'Club', 'Club_Position', 'Preffered_Foot', 'Work_Rate'];
label = 'Rating';

def soccer_player_classification( test_size ):
    data = pan.read_csv(filename);
    data = data.drop( columns = attributes_to_be_removed );

    y = data.loc[:, 'Rating'];
    data = data.drop(columns=label);
    dummieddata = pan.get_dummies(data, columns = categorical_attributes);
    x = dummieddata.iloc[:, :]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, shuffle= False);
    algorihtm = DecisionTreeClassifier();
    X_train_array = num.array(X_train);
    X_test_array = num.array(X_test);
    Y_train_array = num.array(y_train).ravel();
    Y_test_array = num.array(y_test).ravel();

    algorihtm.fit( X_train_array, Y_train_array);
    results = list(algorihtm.predict(X_test_array));
    unique_result = num.unique(results);

    testDict = {}
    predictedDict = {}

    for x in unique_result:
        countTest = 0
        countPredicted = 0
        for y in Y_test_array:
            if x == y:
                countTest+=1
        testDict[x] = countTest
        for z in results:
            if x == z:
                countPredicted+=1
        predictedDict[x] = countPredicted

    test_data_class_name = list(testDict.keys())
    test_data_count = list(testDict.values())

    predicted_data_count = list(predictedDict.values())

    #bar graph
    barWidth = 0.3
    r1 = num.arange(len(test_data_count))
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, test_data_count, color='#3385FF', width=barWidth, edgecolor='white', label='Test')
    plt.bar(r2, predicted_data_count, color='#FF6A33', width=barWidth, edgecolor='white', label='Predicted')
    plt.xlabel('Class', fontweight='bold')
    plt.ylabel('Class Count', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(test_data_class_name))], test_data_count);
    plt.legend()
    plt.show()

    #Line graph
    df = pan.DataFrame({'x': range(len(test_data_count)), 'y1': test_data_count, 'y2': predicted_data_count})
    plt.plot(test_data_class_name, test_data_count, marker='o', markerfacecolor='blue', markersize=6, color='skyblue', linewidth=2, label='Test')
    plt.plot(test_data_class_name, predicted_data_count, marker='o', markerfacecolor='orange', markersize=6, color='orange', linewidth=2, label='Predicted')
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(Y_test_array, results, labels=unique_result)
    #print(pan.DataFrame(cm, index=unique_result, columns=unique_result))

    ax = plt.subplot()
    sns.heatmap(cm, cmap='BuGn', annot=True, fmt='d', ax=ax);
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(unique_result);
    ax.yaxis.set_ticklabels(unique_result);
    plt.show();
    # Confusion Matrix

    print('Accuracy is: ',round(accuracy_score(Y_test_array, results, normalize=True)*100, 3))

    print('Prediction complete')

#This method will classify a single player into one of the 5 classes. Will have to write the code for this.
def classify_player(PlayerName):
     player_input_data = pan.read_csv(input_filename);
     data = pan.read_csv(filename);
     player_input_data = player_input_data.loc[player_input_data['Name'] == PlayerName];
     attributes_to_be_removed = ['Name', 'Preffered_Position', 'Contract_Expiry'];
     categorical_attributes = ['Club_Kit', 'Nationality', 'Club', 'Club_Position', 'Preffered_Foot', 'Work_Rate'];
     label = 'Rating';
     data = data.drop(columns=attributes_to_be_removed);
     player_input_data = player_input_data.drop( columns= attributes_to_be_removed );
     y = data.loc[:, 'Rating'];

     data = data.drop(columns=label);
     dummieddata = pan.get_dummies(data, columns=categorical_attributes);
     dummied_playerinputdata = pan.get_dummies(player_input_data, columns=categorical_attributes);
     input_columns = list(dummieddata.columns);
     input_columns_players = list(dummied_playerinputdata.columns);
     input_columns_players[39] = input_columns_players[39] + '.0';

     input_columns_dict = {};
     for column_name in input_columns:
            input_columns_dict[column_name] = 0;
     for input_column in input_columns_dict:
         if input_columns_players.__contains__(input_column) != True:
             dummied_playerinputdata.insert(loc= len(dummied_playerinputdata.columns),column=input_column, value= input_columns_dict[input_column] );

     x = dummieddata.iloc[:, :]
     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, shuffle=False);
     algorihtm = DecisionTreeClassifier();
     X_train_array = num.array(X_train);
     Y_train_array = num.array(y_train).ravel();

     player_input_array = num.array(dummied_playerinputdata);

     algorihtm.fit(X_train_array, Y_train_array);
     results = list(algorihtm.predict(player_input_array));
     unique_result = num.unique(results);
     return unique_result[0], PlayerName;

def recommender(rating, player_name):
    #read file
    player_input_data = pan.read_csv(input_filename);
    all_players_data = pan.read_csv(filename);

    #pick rows with same rating
    data_for_same_rating = all_players_data.loc[all_players_data['Rating'] == rating];

    #create data frame
    player_input_dataframe = pan.DataFrame(player_input_data)
    data_for_same_rating_dataframe = pan.DataFrame(data_for_same_rating)

    player_input_dataframe = player_input_dataframe.loc[player_input_dataframe['Name'] == player_name];

    #create list using Club column
    list_of_clubs = data_for_same_rating_dataframe['Club'].to_list()
    input_club = player_input_dataframe['Club'].values[0]

    #remove current club from the list
    if(input_club in list_of_clubs):
        indexNames = data_for_same_rating_dataframe[data_for_same_rating_dataframe['Club'] == input_club].index
        data_for_same_rating_dataframe.drop(indexNames, inplace=True)



    player_input_data_filtered = player_input_dataframe.drop(columns=attributes_to_be_removed);
    all_players_data_filtered = data_for_same_rating_dataframe.drop(columns=attributes_to_be_removed);

    player_input_dataframe['mean'] = player_input_data_filtered.mean(axis=1)
    data_for_same_rating_dataframe['mean'] = all_players_data_filtered.mean(axis=1)

    print('Filtered clubs:\n ', data_for_same_rating_dataframe[['Club','mean']])

    input_player_mean = player_input_dataframe['mean'].values[0]
    recommendation_limit = 5

    while True:
        recommended_clubs = data_for_same_rating_dataframe
        recommended_clubs = data_for_same_rating_dataframe.iloc[
            (data_for_same_rating_dataframe['mean'] - input_player_mean).abs().argsort()[:recommendation_limit]]
        recommended_clubs_unique = num.unique(recommended_clubs['Club'].to_list())
        club_count = len(recommended_clubs_unique)
        recommendation_limit+=1
        if club_count>=5:
            break


    print('Predicted class of the Player is : ', rating);
    print('Player details: ', player_input_dataframe['Club'].values[0],' - ', player_input_dataframe['mean'].values[0])
    print('\n',recommended_clubs_unique)
    print('Above are the recommended clubs for: ', player_input_dataframe['Name'].values[0])

def main():
    soccer_player_classification(test_size=0.2);
    result,player_name = classify_player('Lionel Messi');
    #result,player_name = classify_player('Dele Alli');
    #result,player_name = classify_player('Cameron Porter');

    recommender(result, player_name);

if __name__ == "__main__":
    main()