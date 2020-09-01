Required Files:

1. SoccerPlayerStatistics.csv
2. PlayerInputInformation_multiple.csv

To see individual module results:

	1. Classification and Prediction code execution:

		a. Open the classifier.py file and in the main method comment all statements except 'soccer_player_classification(test_size=0.2);'
		b. In a few moments you should be able to see graphs and results.

	2. Recommendation System:

		a. Open the classifier.py file and in the main method : 
			a. comment 'soccer_player_classification(test_size=0.2);' 
			b. uncomment 'result,player_name = classify_player('Lionel Messi');'
			c. uncomment 'recommender(result, player_name);'