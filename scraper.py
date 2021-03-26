from bs4 import BeautifulSoup
import pandas as pd
import requests

def get_scores(year):
    page = requests.get(f"https://www.landofbasketball.com/results/{year}_{year+1}_scores_full.htm")
    soup = BeautifulSoup(page.text, "html.parser")

    left_teams = soup.find_all(class_="left a-left a-right-sm")
    right_teams = soup.find_all(class_="left right-sm")
    home_teams = soup.find_all(class_="right a-right a-left-sm")
    home_points = soup.find_all(class_="left a-right")
    away_points = soup.find_all(class_ = "left right-sm a-right a-left-sm")
    box_scores = [game.a['href'] for game in home_points]

    home = []
    away = []
    for team1, team2, home_team in zip(left_teams,right_teams,home_teams):
        home_team, away_team = (team1.text.strip(), team2.text.strip()) if team1.text in home_team.text else (team1.text.strip(), team2.text.strip())
        home.append(home_team)
        away.append(away_team)


    df = pd.DataFrame({'home': [], 'away': [], 'home_points': [], 'away_points': []})
    df['home'] = pd.Series(home)
    df['away'] = pd.Series(away)
    df['home_points'] = pd.Series([tag.a.text.strip() for tag in home_points])
    df['away_points'] = pd.Series([tag.a.text.strip() for tag in away_points])

    df.to_csv(f"./data/game_scores/{year}_{year+1}_scores.csv")

for year in range(2011,2020):
    get_scores(year)

def get_box(game_url):

    home = pd.DataFrame({'player': [], 'min': [], 'pts': [], 'fg': [], 'fga': [], '3p': [], '3pa': [], 'ft': [], 'fta': [],  'reb': [], 'ast': [], 'stl': [], 'blk': [], 'pf': [], '+/-': [], })
    away = pd.DataFrame({'player': [], 'min': [], 'pts': [], 'fg': [], 'fga': [], '3p': [], '3pa': [], 'ft': [], 'fta': [],  'reb': [], 'ast': [], 'stl': [], 'blk': [], 'pf': [], '+/-': [], })

    pass

