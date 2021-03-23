import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import matplotlib.pyplot as plt

# GAMES_FILE_NAME = 'october_games_19-20'
GAMES_FILE_NAME = 'nov-jan_nba_games'
TEST_GAMES_FILE_NAME = 'feb_nba_games'
TEAMS_FILE_NAME = 'nba_teams_19-20'

USE_COLUMNS = ['PTS/GM', 'aPTS/GM', 'PTS DIFF', 'PACE', 'OEFF', 'DEFF', 'EDIFF', 'SOS', 'rSOS', 'SAR', 'CONS', 'A4F', 'W', 'L', 'WIN%', 'eWIN%', 'pWIN%', 'ACH', 'STRK']

def file_name_to_path(file_name, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(file_name)))

def get_games_data(file_name):

    df = pd.read_csv(file_name_to_path(file_name),
        usecols=['Away Team', 'Away Points', 'Home Team', 'Home Points'])

    return df

def get_teams_data(file_name):

    use_cols = USE_COLUMNS.copy()
    use_cols.append('Team')

    df = pd.read_csv(file_name_to_path(file_name),
        usecols=use_cols)

    return df

def get_x_y(games_file_name, teams_file_name):
    
    # get games data
    games_df = get_games_data(games_file_name)
    # get teams data
    teams_df = get_teams_data(teams_file_name)

    home_df = pd.DataFrame(columns=teams_df.columns)
    away_df = pd.DataFrame(columns=teams_df.columns)

    empty_arr =  np.zeros(2)
    y = np.array([ empty_arr ])
    
    for game_index in range(games_df.shape[0]):
        game_row = games_df.iloc[game_index,:]

        data = [game_row['Home Points'], game_row['Away Points']]
        y = np.concatenate((y, [data]), axis=0)

        for team_index in range(teams_df.shape[0]):
            team_row = teams_df.iloc[team_index,:]

            short_team_name = team_row['Team']

            if short_team_name in game_row['Home Team']:
                home_df.loc[game_index] = team_row
            if short_team_name in game_row['Away Team']:
                away_df.loc[game_index] = team_row

    home_df = home_df.drop(columns=['Team'])
    away_df = away_df.drop(columns=['Team'])

    home_arr = home_df.to_numpy()
    away_arr = away_df.to_numpy()

    empty_arr = np.zeros(len(USE_COLUMNS) * 2)
    x = np.array([ empty_arr ])

    for i in range(home_arr.shape[0]):
        data = np.concatenate((home_arr[i], away_arr[i]), axis=0)
        x = np.concatenate((x, [data]), axis=0)
    
    x = x[1:]
    y = y[1:]

    return (x, y)

def test_run():

    x, y = get_x_y(games_file_name=GAMES_FILE_NAME,
        teams_file_name=TEAMS_FILE_NAME)

    # print('====================')
    # print()
    # print('x: ', x)
    # print()
    # print('--------------------')
    # print()
    # print('y: ', y)
    # print()
    # print('====================')

    model = LinearRegression(normalize=True).fit(x, y)

    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    
    # print('slope:', model.coef_)
    # print('intercept:', model.intercept_)

    test_x, test_y = get_x_y(games_file_name=TEST_GAMES_FILE_NAME,
        teams_file_name=TEAMS_FILE_NAME)
  
    y_pred = model.predict(test_x)

    print('prediction score: ', model.score(test_x, test_y))
    
    data_length = test_y.shape[0]
    x_range = range(data_length)

    test_y = test_y.flatten()
    test_home = test_y[:data_length]
    test_away = test_y[data_length:]

    y_pred = y_pred.flatten()
    home_pred = y_pred[:data_length]
    away_pred = y_pred[data_length:]
    
    plt.plot(x_range, test_home, 'ro', label='Actual Home')
    plt.plot(x_range, test_away, 'bo', label='Actual Away')

    plt.plot(x_range, home_pred, 'yo', label='Predicted Home')
    plt.plot(x_range, away_pred, 'go', label='Predicted Away')

    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    test_run()

