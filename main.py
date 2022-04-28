import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import sys

#pip install, xlrd, numpy, pandas, matplotlib, seaborn

if __name__ == '__main__':

    #Step 1: Define the Problem

    """For this project i want to develop an algorithm that predicts the impact of a player based only in two stats KDR and DMR. 
    I also want to display various stats and get a direct correlation between them
    """

    #Step 2 Finding the dataset (https://www.kaggle.com/gabrieltardochi/counter-strike-global-offensive-matches)

    #Step 3 Import libraries
    import sys  # access to system parameters https://docs.python.org/3/library/sys.html
    print("Python version: {}".format(sys.version))
    import \
        pandas as pd  # collection of functions for data processing and analysis modeled after R dataframes with SQL like features
    print("pandas version: {}".format(pd.__version__))
    import matplotlib  # collection of functions for scientific and publication-ready visualization
    print("matplotlib version: {}".format(matplotlib.__version__))
    import numpy as np  # foundational package for scientific computing
    print("NumPy version: {}".format(np.__version__))
    import scipy as sp  # collection of functions for scientific computing and advance mathematics
    print("SciPy version: {}".format(sp.__version__))
    import IPython
    from IPython import display  # pretty printing of dataframes (Only for testing in Jupiter notebook)
    print("IPython version: {}".format(IPython.__version__))
    import sklearn  # collection of machine learning algorithms
    print("scikit-learn version: {}".format(sklearn.__version__))
    # misc libraries
    import random
    import time

    # Common Model Algorithms
    from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, \
        gaussian_process
    from xgboost import XGBClassifier

    # Common Model Helpers
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn import feature_selection
    from sklearn import model_selection
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    # Visualization
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import seaborn as sns
    from pandas.plotting import scatter_matrix

    # Configure Visualization Defaults
    mpl.style.use('ggplot')
    sns.set_style('white')
    pylab.rcParams['figure.figsize'] = 12, 8
    matplotlib.rcParams['figure.figsize'] = (20, 60)

    sns.set_theme(rc={'grid.linewidth': 0.5,
                      'axes.linewidth': 0.75, 'axes.facecolor': '#fff3e9', 'axes.labelcolor': '#6b1000',
                      'figure.facecolor': '#f7e7da',
                      'xtick.labelcolor': '#6b1000', 'ytick.labelcolor': '#6b1000'})

    # import data from file: https://www.kaggle.com/gabrieltardochi/counter-strike-global-offensive-matches
    data_raw = pd.read_csv('venv/input/train.csv')

    # broken into 3 splits: train, test, and validation
    data_val = pd.read_csv('venv/input/test.csv')

    # to play with our data we'll create a copy
    # python assignment or equal passes by reference vs values, so I used the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
    data1 = data_raw.copy(deep=True)

    # passing by reference so I can clean both datasets at once
    data_cleaner = [data1, data_val]

    # preview data
    print(data_raw.info())  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
    # data_raw.head() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html
    # data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html
    data_raw.sample(10)  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html

    #Correcting, Completing, Creating, and Converting
    #pd.set_option('display.max_rows', None) #Let us see all columns in console
    print('Train columns with null values:\n', data1.isnull().sum())
    print("-" * 10)
    print('Test/Validation columns with null values:\n', data_val.isnull().sum())
    print("-" * 10)
    pd.set_option('display.max_rows', 5) #Reset console display format
    data_raw.describe(include='all')

    ###COMPLETING: complete or delete missing values in train and test/validation dataset
    for dataset in data_cleaner:
        # complete missing t1_player5_wins_perc_after_fk with median
        dataset['t1_player5_wins_perc_after_fk'].fillna(dataset['t1_player5_wins_perc_after_fk'].median(), inplace=True)

        # complete t2_player3_wins_perc_after_fk with mode
        dataset['t2_player3_wins_perc_after_fk'].fillna(dataset['t2_player3_wins_perc_after_fk'].mode()[0], inplace=True)

        # complete missing t2_player5_wins_perc_after_fk with median
        dataset['t2_player5_wins_perc_after_fk'].fillna(dataset['t2_player5_wins_perc_after_fk'].median(), inplace=True)

        #Change winner values from team1 and team2 to 1 and 2 (Replaced, Keeping the comment just to renember)
        #dataset.loc[(dataset.winner == 't1'),'winner']=1
        #dataset.loc[(dataset.winner == 't2'), 'winner'] = 2

        #Since we have stats for each player, lets replace individual stats with team stats using mean.

        #rating
        dataset['mean_rating_team1'] = dataset.iloc[:, [10, 26, 42, 58, 74]].mean(axis=1)
        dataset['mean_rating_team2'] = dataset.iloc[:, [90, 106, 122, 138, 154]].mean(axis=1)

        #impact
        dataset['mean_impact_team1'] = dataset.iloc[:, [11, 27, 43, 59, 75]].mean(axis=1)
        dataset['mean_impact_team2'] = dataset.iloc[:, [91, 107, 123, 139, 155]].mean(axis=1)

        #kdr
        dataset['mean_kdr_team1'] = dataset.iloc[:, [12, 28, 44, 60, 76]].mean(axis=1)
        dataset['mean_kdr_team2'] = dataset.iloc[:, [92, 108, 124, 140, 156]].mean(axis=1)

        # dmr
        dataset['mean_dmr_team1'] = dataset.iloc[:, [13, 29, 45, 61, 77]].mean(axis=1)
        dataset['mean_dmr_team2'] = dataset.iloc[:, [93, 109, 125, 141, 157]].mean(axis=1)

        #kpr
        dataset['mean_kpr_team1'] = dataset.iloc[:, [14, 30, 46, 62, 78]].mean(axis=1)
        dataset['mean_kpr_team2'] = dataset.iloc[:, [94, 110, 126, 142, 158]].mean(axis=1)

        #apr
        dataset['mean_apr_team1'] = dataset.iloc[:, [15, 31, 47, 63, 79]].mean(axis=1)
        dataset['mean_apr_team2'] = dataset.iloc[:, [95, 111, 127, 143, 159]].mean(axis=1)

        #dpr
        dataset['mean_dpr_team1'] = dataset.iloc[:, [16, 32, 48, 64, 80]].mean(axis=1)
        dataset['mean_dpr_team2'] = dataset.iloc[:, [96, 112, 128, 144, 160]].mean(axis=1)

        # spr
        dataset['mean_spr_team1'] = dataset.iloc[:, [17, 33, 49, 65, 81]].mean(axis=1)
        dataset['mean_spr_team2'] = dataset.iloc[:, [97, 113, 129, 145, 161]].mean(axis=1)

        # opk ratio
        dataset['mean_opk_team1'] = dataset.iloc[:, [18, 34, 50, 66, 82]].mean(axis=1)
        dataset['mean_opk_team2'] = dataset.iloc[:, [98, 114, 130, 146, 162]].mean(axis=1)

        #Clutch_win_perc +7 rows
        dataset['mean_clutch_win_perc_team1'] = dataset.iloc[:, [25, 41, 57, 73, 89]].mean(axis=1)
        dataset['mean_clutch_win_perc_team2'] = dataset.iloc[:, [101, 121, 137, 153, 169]].mean(axis=1)

        #Create a win_team1 and win_team2 where o is no and 1 is yes so we can compare values in correlation

        dataset.loc[dataset['winner'] == "t1", '"team_1_win'] = 1
        dataset.loc[dataset['winner'] == "t1", '"team_2_win'] = 0
        dataset.loc[dataset['winner'] == "t2", '"team_1_win'] = 0
        dataset.loc[dataset['winner'] == "t2", '"team_2_win'] = 1
        #Teams can draw so it will count as a lose
        dataset.loc[dataset['winner'] == "draw", '"team_1_win'] = 0
        dataset.loc[dataset['winner'] == "draw", '"team_2_win'] = 0


    pd.set_option('display.max_rows', None)  # Let us see all columns in console
    print(data1.isnull().sum())
    print("-" * 10)
    print(data_val.isnull().sum())
    pd.set_option('display.max_rows', 5)  # Let us see default columns in console


    # CONVERT: convert objects to category using Label Encoder for train and test/validation dataset

    # code categorical data (It is coded manually before, as team_1_win, but this time we replace the original value instead of adding a new column
    label = LabelEncoder()
    for dataset in data_cleaner:
        dataset['winner'] = label.fit_transform(dataset['winner'])


    #--------------------Visualization---------------------------

    #Create a new dataframe only with mean values to analize team performance
    dataTeamMeans = data1.copy(deep=True)
    dataTeamMeans.drop(dataTeamMeans.columns[10:170], axis = 1, inplace = True)
    dataTeamMeans.drop(dataTeamMeans.columns[0:9], axis=1, inplace=True)
    dataTeamMeans.round(decimals=2)

    def correlation_heatmap(df):
        _, ax = plt.subplots(figsize=(14, 12))
        colormap = sns.diverging_palette(220, 10, as_cmap=True)
        _ = sns.heatmap(
            df.corr(),
            cmap=colormap,
            square=True,
            cbar_kws={'shrink': .9},
            ax=ax,
            annot=True,
            linewidths=0.1, vmax=1.0, linecolor='white',
            annot_kws={'fontsize': 8}
        )

        plt.title('Correlation of Features', y=1.05, size=15)


    correlation_heatmap(dataTeamMeans) #Show correlation map
    figure = plt.gcf()
    figure.set_size_inches(20, 11.25) #To change size we need to get the current figure
    plt.savefig('Correlation.jpg')


    #Check how Damage per round affect the impact
    sns.lmplot(x="mean_dmr_team1", y="mean_impact_team1", data=dataTeamMeans, fit_reg=True)
    figure = plt.gcf()
    figure.set_size_inches(20, 5) #To change size we need to get the current figure
    plt.savefig('Relation_DamageVSImpact.jpg')

    #Check how kills per round affect the impact
    sns.lmplot(x="mean_kpr_team1", y="mean_impact_team1", data=dataTeamMeans, fit_reg=True)
    figure = plt.gcf()
    figure.set_size_inches(20, 5) #To change size we need to get the current figure
    plt.savefig('Relation_KillsVSImpact.jpg')

    # Check how assists per round affect the impact
    sns.lmplot(x="mean_apr_team1", y="mean_impact_team1", data=dataTeamMeans, fit_reg=True)
    figure = plt.gcf()
    figure.set_size_inches(20, 5) #To change size we need to get the current figure
    plt.savefig('Relation_AssistVSImpact.jpg')

    #For player stats I will only check the top player for each match since hes the most impactful (based on how data is distributed in the dataset)
    #Player1 always will be the "best player" for each match

    # Check how clutch percentage affects the impact
    sns.lmplot(x="t1_player1_clutch_win_perc", y="t1_player1_impact", data=data1, fit_reg=True)
    figure = plt.gcf()
    figure.set_size_inches(20, 5) #To change size we need to get the current figure
    plt.savefig('Relation_ClutchVSImpact_player.jpg')

    # Check how multikill percentage affects the impact
    sns.lmplot(x="t1_player1_multikill_perc", y="t1_player1_impact", data=data1, fit_reg=True)
    figure = plt.gcf()
    figure.set_size_inches(20, 5) #To change size we need to get the current figure
    plt.savefig('Relation_MultikillSImpact_player.jpg')


    #Check how opening kills affects the impact
    sns.lmplot(x="t1_player1_opk_ratio", y="t1_player1_impact", data=data1, fit_reg=True)
    figure = plt.gcf()
    figure.set_size_inches(20, 5) #To change size we need to get the current figure
    plt.savefig('Relation_OpkVSImpact_player.jpg')

    #How many teams were top20
    # Get unique elements in multiple columns team_1 and team_2
    column_values = data1[["team_1", "team_2"]].values.ravel()
    uniqueTeams = pd.unique(column_values)
    #print(uniqueTeams)
    #print(len(uniqueTeams))    Shows hoy many uniqe teams were in top 20

    #How many matches were played by each team
    #Since a team can appear in team_1 or team_2 column we need to count in both columns

    teamsCombined = data1[['team_1', 'team_2']].apply(pd.Series.value_counts)  #Generate a counter for each value in team_1 and team_2 column
    sum_teams = teamsCombined["team_1"].fillna(0) + teamsCombined["team_2"].fillna(0)           #Add team_1 and team_2 ocurrencies fill nulls with 0
    sum_teams = sum_teams.sort_values(ascending=False)
    sum_teams.plot.bar(rot=0)   #Generate a plot bat with teams and times they played in top 20
    #plt.figure(figsize=(8, 6), dpi=80)
    plt.xticks(rotation=90)
    figure = plt.gcf()
    figure.set_size_inches(20, 8) #To change size we need to get the current figure
    plt.savefig('teamsMatchesCount.jpg', bbox_inches='tight')

    #--------------------------training and prediction --------------------------

    x = data1[["t1_player1_kdr","t1_player1_dmr"]]
    y = data1["t1_player1_impact"]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

    clf = LinearRegression()
    clf.fit(x_train, y_train)

    print(clf.predict(x_test))
    print(clf.score(x_test, y_test))

    #Not able to predict player impact outcome based on those 2 inputs alone (53%precission)

    #-------------------------Display---------------------------------

    #I've been unable to display the data properly due to incompabilities with my system, so I decided to save the images instead
    # locally.

    """
    Como conclusión, se puede confirmar que el impacto de un jugador no se puede medir tan solo con su daño por ronda y 
    bajas por ronda.
    
    No osbtante en la tabla de correlations, se puede observar como el Daño por ronda y las bajas por ronda tienen un 
    impacto significativo para conseguir una victoria.
    
    Las asistencias también tienen una relacion directa a la hora de conseguir una victoria pero en menos medida (0.14 vs 0.12)
    
    En cuanto a los plot charts y las regresiones lineales:
        
        Es interesante observar como el clutch rate vs Player_impact de los jugadores tiene un slope bastante reducido en comparación
        a las otras gráficas.
        Sin embargo cuando sacamos la media de los clutches de todos los jugadores del equipo se observa como tiene un 0.19 de correlation
        con la media del impacto de equipo, pero no tiene una relacion directa con el winning outcome del mismo.
        
        La explicación puede ser que los clutches no están tan concentrados en un solo jugador, o estos se dan en pocas ocasiones durante
        las partidas con lo cual analizarlos individualmente es poco significativo.
        
    Se observa como las asistencias tienen un impacto muchisimo menor en comparación a los kills, lo cual era de esperar.
    
    Uno de los resultados que no esperaba es que el slope de Multikill sea superior al de OPK (Openning Kills).
        Esto es debido a que es común la creencia de que una de las mejores formas de ganar ventaja en una ronda es conseguir la primera baja,
        sin embargo las gráfica muestra que tiene mucho mas impacto conseguir una baja multiple.
        (Hay que tener en cuenta que dentro de la estadística de baja múltiple puede recoger también el opennig kill, es decir puedes 
        conseguir la primera baja y además acabar con otro/otros jugadores mas"
        
    ------------------------------------------------------------------English----------------------------------------------------------------------
        
    As a conclusion, it can be confirmed that the impact of a player cannot be measured only with his damage per round and
    kills per round.
    
    In the correlation table, it can be seen how the Damage per round and the losses per round have a
    significant impact to achieve a victory.
    
    Assists also have a direct relationship when it comes to achieving a victory but to a lesser extent (0.14 vs 0.12)
    
    Regarding plot charts and linear regressions:
        
        It is interesting to observe how the clutch rate vs Player_impact of the players has a fairly low slope in comparison
        to the other graphs.
        However, when we take the average of the clutches of all the players on the team, it is observed that it has a 0.19 correlation
        with the average of the team impact, but it does not have a direct relationship with the winning outcome.
        
        The explanation may be that the clutches are not as concentrated in a single player, or they are rarely given during
        the games with which to analyze them individually is not very significant.
        
    It is observed how the assists have a much lower impact compared to the kills, which was to be expected.
    
    One of the results that I did not expect is that the slope of Multikill is higher than that of OPK (Openning Kills).
        This is due to the common belief that one of the best ways to gain an advantage in a round is to get the first kill,
        however, the graphs show that it has much more impact to get a multiple kill.
        (Keep in mind that within the multiple kill statistic you can also collect the opennig kill, that is, you can
        get the first kill and also kill another player after "
   
    """























