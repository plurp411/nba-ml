import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
import time

PLAYERS_FILE_NAME = 'some_players'
GAMES_FILE_NAME = 'october_games_19-20'

PLAYERS_USE_COLUMNS = ['minutes', 'FG', '3P', 'FT', 'O', 'D', 'Reb', 'Ast', 'Stl', 'Blk', 'TO', 'PF', '+/-', 'Pts']
# GAMES_USE_COLUMNS = ['Away Team', 'Away Points', 'Home Team', 'Home Points']

def file_name_to_path(file_name, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(file_name)))

def get_dataframe(file_name, use_columns, append_columns=[], time_columns=[]):

    use_cols = use_columns.copy()

    for append_column in append_columns:    
        use_cols.append(append_column)

    df = pd.read_csv(file_name_to_path(file_name),
        parse_dates=time_columns,
        usecols=use_cols)

    return df

def get_x_y(players_file_name, games_file_name):
    
    # get players dataframe
    temp_players_df = get_dataframe(file_name=players_file_name,
        use_columns=PLAYERS_USE_COLUMNS,
        append_columns=['player'],
        time_columns=['minutes'])
    # get games dataframe
    # games_df = get_dataframe(file_name=games_file_name,
    #     use_columns=GAMES_USE_COLUMNS)
    







    # convert minutes played type to float
    for i in range(temp_players_df['minutes'].shape[0]):

        timestamp_str = temp_players_df['minutes'].iloc[i]
        timestamp_str = timestamp_str[:-3]

        timestamp_str = time.strptime(timestamp_str, '%M:%S')

        minutes = datetime.timedelta(hours=timestamp_str.tm_hour,
            minutes=timestamp_str.tm_min,
            seconds=timestamp_str.tm_sec).total_seconds() / 60.0

        temp_players_df['minutes'].iloc[i] = minutes

    # divide stats byminutes played
    temp_players_df.iloc[:, 5:] = temp_players_df.iloc[:, 5:].div(temp_players_df['minutes'], axis=0)








    # get all player names
    player_names_arr = temp_players_df['player'].to_numpy()
    # remove duplicates
    player_names_arr = np.unique(player_names_arr)

    # create empty df for players means
    players_mean_df = pd.DataFrame(index=player_names_arr, columns=PLAYERS_USE_COLUMNS)

    # loop over player names
    for player_name in player_names_arr:

        # select rows that match player name
        player_df = temp_players_df.loc[temp_players_df['player'] == player_name]
        # take mean of matched rows
        player_mean_df = player_df.mean()

        # assign players mean in dataframe
        players_mean_df.loc[player_name] = player_mean_df

    # fill nans with 0s
    players_mean_df = players_mean_df.fillna(0.0)








    print(players_mean_df)













    # home_df = pd.DataFrame(columns=teams_df.columns)
    # away_df = pd.DataFrame(columns=teams_df.columns)

    # empty_arr = np.zeros(2)
    # y = np.array([ empty_arr ])
    
    # for game_index in range(games_df.shape[0]):
    #     game_row = games_df.iloc[game_index,:]

    #     data = [game_row['Home Points'], game_row['Away Points']]
    #     y = np.concatenate((y, [data]), axis=0)

    #     for team_index in range(teams_df.shape[0]):
    #         team_row = teams_df.iloc[team_index,:]

    #         short_team_name = team_row['Team']

    #         if short_team_name in game_row['Home Team']:
    #             home_df.loc[game_index] = team_row
    #         if short_team_name in game_row['Away Team']:
    #             away_df.loc[game_index] = team_row

    # home_df = home_df.drop(columns=['Team'])
    # away_df = away_df.drop(columns=['Team'])






    # category = 'OEFF'
    
    # home_df[category] = home_df[category] / away_df[category]
    # x_points = home_df[category].to_numpy()
    # games_df['Home Points'] = games_df['Home Points'] / games_df['Away Points']
    # y_points = games_df['Home Points'].to_numpy()

    # plt.plot(x_points, y_points, 'yo', label='Data')

    # plt.title('Relationship Correlation')
    # plt.xlabel('Points / Game')
    # plt.ylabel('Points Scored')

    # plt.legend(loc='upper left')
    # plt.show()






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

    x, y = get_x_y(players_file_name=PLAYERS_FILE_NAME,
        games_file_name=GAMES_FILE_NAME)

    # print('====================')
    # print()
    # print('x: ', x)
    # print()
    # print('--------------------')
    # print()
    # print('y: ', y)
    # print()
    # print('====================')

    model = LinearRegression(normalize=True)
    # model = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    # model = RandomForestRegressor()

    model.fit(x, y)

    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    
    # print('slope:', model.coef_)
    # print('intercept:', model.intercept_)

    test_x, test_y = get_x_y(games_file_name=TEST_GAMES_FILE_NAME,
        teams_file_name=TEAMS_FILE_NAME)
  
    y_pred = model.predict(test_x)

    print('prediction score: ', model.score(test_x, test_y))
    
    # data_length = test_y.shape[0]
    # x_range = range(data_length)

    # test_y = test_y.flatten()
    # test_home = test_y[:data_length]
    # test_away = test_y[data_length:]

    # y_pred = y_pred.flatten()
    # home_pred = y_pred[:data_length]
    # away_pred = y_pred[data_length:]
    
    # plt.plot(x_range, test_home, 'ro', label='Actual Home')
    # plt.plot(x_range, test_away, 'bo', label='Actual Away')

    # plt.plot(x_range, home_pred, 'yo', label='Predicted Home')
    # plt.plot(x_range, away_pred, 'go', label='Predicted Away')

    # plt.title(f'Model: {GAMES_FILE_NAME}.csv, Test: {TEST_GAMES_FILE_NAME}.csv')
    # plt.xlabel('Game Count')
    # plt.ylabel('Points Scored')

    # plt.legend(loc='upper left')
    # plt.show()

if __name__ == "__main__":
    test_run()

