# coding: utf-8
import numpy as np
import pandas as pd

# for own libs
import sys
sys.path.append('//')

# lib for working with SQL
import data_load as dl

# for graphics
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

# result of match as label and as number
score_dct={
    'Win': 1,
    'Lose': 0,
    'Draw': 0.5,
    'Unknown': 0
        }

# result of match as label and as number
score_label_dct={
    'Win': '2.Win',
    'Lose': '0.Lose',
    'Draw': '1.Draw',
        }

# every league has own started rating
league_rating_dct = {
    # Spain
    140: 2400, # La liga
    141: 2200, # Segunda
}

# get proper dataframe format
def formatting(df, gh, ga):
    # date is string by default
    df['date']=df['date'].astype('datetime64[ns]')
    
    # -1 vs NULL when there are no game result
    df['home_result']=np.where(
        (df[gh]>df[ga]), 'Win', 
                                    np.where(
                                        (df[gh]<df[ga]), 'Lose', 'Draw'
                                            )
                                )
    df['away_result']=np.where(
        (df[gh]>df[ga]), 'Lose', 
                                    np.where(
                                        (df[gh]<df[ga]), 'Win', 'Draw'
                                            )
                                )
    
    df=df.rename(columns={
                        gh: 'home_score',
                        ga: 'away_score',
                        'home': 'home_team',
                        'away': 'away_team'
                          })

    
    return df

# dataframe when team in focus
# 2 raws in every match
def team_table(df):
    # first of all tale only home teams 
    df_home=df.copy()

    df_home['team'] = df_home['home_team']
    df_home['opponent'] = df_home['away_team']
    df_home['result'] = df_home['home_result']
    df_home['team_score'] = df_home['home_score']
    df_home['opponent_score'] = df_home['away_score']
    df_home['venue'] = 'Home'
    df_home['team_id'] = df_home['home_id']
    df_home['opponent_team_id'] = df_home['away_id']

    df_home=df_home[[
        'match_id', 'date',
        'team', 'opponent', 'venue', 
        'result', 
        'team_score', 'opponent_score',
        'team_id', 'opponent_team_id'
                    ]]
    
    # and away team symmetrical
    df_away=df.copy()
    
    df_away['team'] = df_away['away_team']
    df_away['opponent'] = df_away['home_team']
    df_away['result'] = df_away['away_result']
    df_away['team_score'] = df_away['away_score']
    df_away['opponent_score'] = df_away['home_score']
    df_away['venue'] = 'Away'
    df_away['team_id'] = df_away['away_id']
    df_away['opponent_team_id'] = df_away['home_id']
    
    df_away=df_away[[
    'match_id', 'date',
    'team', 'opponent', 'venue', 
    'result', 
    'team_score', 'opponent_score',
     'team_id', 'opponent_team_id'
                ]]

    # get all tohether
    team_df=pd.concat([df_home, df_away])
    team_df=team_df.sort_values(by='date', ascending=True)
    team_df = team_df.reset_index()
    team_df['game_number']=team_df.sort_values(['date'], ascending=[True]) \
                 .groupby(['team']) \
                 .cumcount() + 1
    team_df['id'] = team_df.index
    
    # get additional fields
    # footbal points
    team_df['points'] = np.where(
                                        team_df['result']=='Win', 3,
                                        np.where(
                                            team_df['result']=='Draw', 1, 0
                                                )
                                      )
    # chess points
    team_df['score'] = np.where(
                                        team_df['result']=='Win', 1,
                                        np.where(
                                            team_df['result']=='Draw', 0.5, 0
                                                )
                                      )
    # usefull bool fields
    result_ohe = pd.get_dummies(team_df['result'], prefix='is')
    team_df = pd.concat([team_df, result_ohe], axis=1)

    # very important sorting
    team_df = team_df.sort_values(
        by=['date', 'game_number', 'venue'], 
        ascending=[True, True, False])

    team_df['is_dry_match'] = np.where(team_df['opponent_score']==0, 1, 0)

    # whole history stat
    team_df['wins'] = team_df.sort_values(by=['date'], ascending=True)\
                           .groupby(['team'])['is_Win']\
                           .cumsum()
    team_df['draws'] = team_df.sort_values(by=['date'], ascending=True)\
                           .groupby(['team'])['is_Draw']\
                           .cumsum()
    team_df['loses'] = team_df.sort_values(by=['date'], ascending=True)\
                           .groupby(['team'])['is_Lose']\
                           .cumsum()

    team_df['scored'] = team_df.sort_values(by=['date'], ascending=True)\
                           .groupby(['team'])['team_score']\
                           .cumsum()
    team_df['mised'] = team_df.sort_values(by=['date'], ascending=True)\
                           .groupby(['team'])['opponent_score']\
                           .cumsum()
    return team_df

# return classic Elo propabilities
# elo_prob(2882, 2722) -> 0.7152 (72% chanses Carlsen (2882) to beat Wan Hao (2722))
def elo_prob(rw, rb):
    try:
        rw=float(rw)
        rb=float(rb)
        res=1/(1+np.power(10, (rb-rw)/400))
    except:
        0.5
    return res

# rating changing after game
def elo_rating_changes(rating, opponent_rating, score, games):
    K = 20
    # fast tunnel for newcomers
#     if games<=30:
#         K=40
#     else:
#         # slow tunnel for tops
#         if rating>2500:
#             K=10
#         elif rating<=2500:
#             K=20
            
    expectation=elo_prob(rating, opponent_rating)
    new_rating=rating+K*(score-expectation)
    
    return np.round(new_rating,2)

# get list of interactions by list
def get_all_sets(ll):
    rr=[]
    r=[]
    for i in ll:
        for j in ll:
            if i!=j:
                rr.append(str(i)+'-'+str(j))
    return rr


def count_elo_rating_series(
                    team_ser,
                    opponent_ser,
                    rating_ser, 
                    opponent_rating_ser, 
                    score_ser, 
                    game_number_ser
                    ):
    # very important dict
    current_rating_dict={}
    # for final return
    rating_lst = []
    opponent_rating_lst = []
    # for noname teams and errors
    default_rating_value = 2400

    # daraframe withought details
    base_stat_df = pd.concat([team_ser, rating_ser], axis=1).drop_duplicates()  
    # teamlines and default ratings
    #current_rating_dict=dict(zip(base_stat_df['team'], base_stat_df['rating']))
    current_rating_dict=dict(zip(base_stat_df[base_stat_df.columns[0]], base_stat_df[base_stat_df.columns[1]]))
    
    for i in range(len(team_ser)):
        # tring to get current rating from dict
        try:
            r1 = current_rating_dict[team_ser.values[i]]
        except:
            r1 = default_rating_value
        # and for opponent    
        try:
            r2 = current_rating_dict[opponent_ser.values[i]]
        except:
            r2 = default_rating_value
        
        # and culc new one
        new_rating=elo_rating_changes(
                                    r1,  
                                    r2,
                                    score_ser.values[i],
                                    game_number_ser.values[i]
                                        )
        
        if new_rating >= 0:
            pass
        else:
            print('this:', i)
        # isnt' correct for necomers and other asymmetrical things
        new_opponent_rating = r2 - (new_rating - r1)
        
        rating_lst.append(new_rating)
        opponent_rating_lst.append(new_opponent_rating)
        
        # match in 2 rows, thats why updrade every 2 times in for
        if (i % 2) == 1:
            current_rating_dict.update({team_ser.values[i]: new_rating})
            current_rating_dict.update({opponent_ser.values[i]: new_opponent_rating})

    return current_rating_dict, rating_lst, opponent_rating_lst

def count_elo_rating_df(df):
    # teams and their league at the start of history
    start_ratings = pd.read_csv('start_ratings.csv')
    start_ratings['start_rating'] = start_ratings['start_rating'].fillna(2286)
    
    df = df.merge(
        start_ratings, 'left', left_on='team_id', right_on='id', suffixes=('', '_rating')
                                        )
    df = df.merge(
        start_ratings, 'left', left_on='opponent_team_id', right_on='id', suffixes=('', '_opponent')
                                        )
    df = df.drop(columns=['team_rating', 'id_rating', 'team_opponent', 'id_opponent'])
    df['start_rating'] = df['start_rating'].fillna(2286)
    df['start_rating_opponent'] = df['start_rating_opponent'].fillna(2286)

    df['rating'] = np.where(
                        df['is_high_league'] == 0, 
                        df['start_rating'] - 200,
                        df['start_rating']
                                        )
    df['opponent_rating'] = np.where(
                        df['is_high_league'] == 0, 
                        df['start_rating_opponent'] - 200,
                        df['start_rating_opponent']
                                        )
    
    df = df.drop(columns=['start_rating', 'start_rating_opponent'])

    # and here we go
    current_rating_dict, rating_lst, opponent_rating_lst = count_elo_rating_series(
                        df['team'],
                        df['opponent'],
                        df['rating'], 
                        df['opponent_rating'], 
                        df['score'], 
                        df['game_number']
                    )
    # history of rating to dataframe
    df['rating'] = rating_lst
    df['opponent_rating'] = opponent_rating_lst
    df['rating_difference'] = df['rating'] - df['opponent_rating']
    
    df['elo_propability']=df[['rating', 'opponent_rating']].apply(
                    lambda x: np.round(
                           elo_prob(
                                    x[0], 
                                    x[1] 
                                    ),2), axis=1)
    
    return df, current_rating_dict

def calc_pers_stat(df):
    curr_stat_dct = {}
    
    lst = []
    n_lst = []
    
    res_lst = []
    
    for i in range(len(df)):
        team = df['team'].values[i]
        opponent = df['opponent'].values[i]
        venue = df['venue'].values[i]
        
        key = team + '-' + opponent
        
        wins = df['is_Win'].values[i]
        draws = df['is_Draw'].values[i]
        loses = df['is_Lose'].values[i]
        
        scored = df['team_score'].values[i]
        missed = df['opponent_score'].values[i]
        
        if venue == 'Home':
                    wins_home = df['is_Win'].values[i]
                    draws_home = df['is_Draw'].values[i]
                    loses_home = df['is_Lose'].values[i]

                    scored_home = df['team_score'].values[i]
                    missed_home = df['opponent_score'].values[i]
        else:
                    wins_home = 0
                    draws_home = 0
                    loses_home = 0

                    scored_home = 0
                    missed_home = 0
                    
        if venue == 'Away':
                    wins_away = df['is_Win'].values[i]
                    draws_away = df['is_Draw'].values[i]
                    loses_away = df['is_Lose'].values[i]

                    scored_away = df['team_score'].values[i]
                    missed_away = df['opponent_score'].values[i]
        else:
                    wins_away = 0
                    draws_away = 0
                    loses_away = 0

                    scored_away = 0
                    missed_away = 0
            
        
        try:
            lst = [
                curr_stat_dct[key][0] + wins,
                curr_stat_dct[key][1] + draws,
                curr_stat_dct[key][2] + loses,
                curr_stat_dct[key][3] + scored,
                curr_stat_dct[key][4] + missed,
                
                curr_stat_dct[key][5] + wins_home,
                curr_stat_dct[key][6] + draws_home,
                curr_stat_dct[key][7] + loses_home,
                curr_stat_dct[key][8] + scored_home,
                curr_stat_dct[key][9] + missed_home,
                
                curr_stat_dct[key][10] + wins_away,
                curr_stat_dct[key][11] + draws_away,
                curr_stat_dct[key][12] + loses_away,
                curr_stat_dct[key][13] + scored_away,
                curr_stat_dct[key][14] + missed_away,
                  ]
        except:
            lst = [
                wins,
                draws,
                loses,
                scored,
                missed,
                
                wins_home,
                draws_home,
                loses_home,
                scored_home,
                missed_home,
                
                wins_away,
                draws_away,
                loses_away,
                scored_away,
                missed_away,
                  ]
            
        curr_stat_dct.update({
                            key : lst
                                })
        res_lst.append(lst)

    return curr_stat_dct, res_lst

def pes_stat_venue(pers_stat_df):
    pers_stat_df.columns = [
                        'wins',
                        'draws',
                        'loses',
                        'scored',
                        'missed',

                        'wins_home',
                        'draws_home',
                        'loses_home',
                        'scored_home',
                        'missed_home',

                        'wins_away',
                        'draws_away',
                        'loses_away',
                        'scored_away',
                        'missed_away',
                        'id',
        ]

    

    pers_stat_df_h = pers_stat_df[[
                    'id',
                    'wins',
                    'draws',
                    'loses',
                    'scored',
                    'missed',

                    'wins_home',
                    'draws_home',
                    'loses_home',
                    'scored_home',
                    'missed_home',
    ]].copy()
    pers_stat_df_h.columns = [
                    'id',
                    'wins_peronal_meet',
                    'draws_peronal_meet',
                    'loses_peronal_meet',
                    'scored_peronal_meet',
                    'missed_peronal_meet',

                    'wins_venue',
                    'draws_venue',
                    'loses_venue',
                    'scored_venue',
                    'missed_venue',
    ]
    pers_stat_df_h['venue'] = 'Home'

    pers_stat_df_a = pers_stat_df[[
                    'id',
                    'wins',
                    'draws',
                    'loses',
                    'scored',
                    'missed',

                    'wins_away',
                    'draws_away',
                    'loses_away',
                    'scored_away',
                    'missed_away',
    ]].copy()
    pers_stat_df_a.columns = [
                    'id',
                    'wins_peronal_meet',
                    'draws_peronal_meet',
                    'loses_peronal_meet',
                    'scored_peronal_meet',
                    'missed_peronal_meet',

                    'wins_venue',
                    'draws_venue',
                    'loses_venue',
                    'scored_venue',
                    'missed_venue',
    ]
    pers_stat_df_a['venue'] = 'Away'

    pers_stat_df_venue = pd.concat([pers_stat_df_h, pers_stat_df_a])
    
    pers_stat_df_venue['result_difference_peronal_meet'] = (pers_stat_df_venue['wins_peronal_meet'] 
                                                        - pers_stat_df_venue['loses_peronal_meet'])
    pers_stat_df_venue['result_difference_venue'] = (pers_stat_df_venue['wins_venue'] 
                                                            - pers_stat_df_venue['loses_venue'])
    pers_stat_df_venue['games_peronal_meet'] = (
                                             pers_stat_df_venue['wins_peronal_meet'] 
                                           + pers_stat_df_venue['loses_peronal_meet'] 
                                           + pers_stat_df_venue['draws_peronal_meet']
                                            )

    pers_stat_df_venue['games_venue'] = (
                                             pers_stat_df_venue['wins_venue'] 
                                           + pers_stat_df_venue['loses_venue'] 
                                           + pers_stat_df_venue['draws_venue']
                                            )

    return pers_stat_df_venue


def calc_pers_stat_df(df):
    curr_stat_dct, res_lst = calc_pers_stat(df)
    pers_stat_df = pd.DataFrame(res_lst)
    pers_stat_df['id'] = df['id'].values
    
    pers_stat_df_venue = pes_stat_venue(pers_stat_df)
        
    df = df.merge(pers_stat_df_venue, 'left', on=['id', 'venue'], suffixes=('', '_personal_meet'))
    
    return df, curr_stat_dct



















# count rating for bi dataframe
def count_elo_rating(team_df):
    # dicts for current ratings, 
    # will be updated after every game
    # no history
    teams=team_df['team'].value_counts().index
    
    # dafault rating is 2400
    # it maybe every number
    default_ratings=np.ones(len(teams))*2400
    current_rating_dict=dict(zip(teams, default_ratings))
    
    # personal meets dict
    current_stat_dict=dict({'team1-team2': np.zeros(5)})

    # results for dataframe
    ratings_lst=[]
    opponent_ratings_lst=[]
    ratings_chng_lst=[]
    wins_lst=[]
    draws_lst=[]
    loses_lst=[]
    scored_lst=[]
    mised_lst=[]
    
    # time sorting is very important
    team_df=team_df.sort_values(by=['date', 'team'])
    
    # big loop
    for match in team_df.values:
        # example of match variable
        # 699 Timestamp('2000-04-15 00:00:00') 'malaysia' 'national' 'Perlis'
        # 'Selangor FC' 'Away' 'Draw' 'F' 2 2 
        # 'data/football-data/data/results/malaysia.csv' 1
        
        # start stat
        wins=0
        draws=0
        loses=0
        scored=0
        mised=0
        
        # if we know result we should count
        if match[7]!='Unknown':
            # teams pair
            curr_t_str=str(match[4])+'-'+str(match[5])
            
            if (curr_t_str not in current_stat_dict.keys()):
                current_stat_dict.update({
                                        curr_t_str:np.zeros(5)
                                            })
            # update all stats
            if match[7]=='Win':
                current_stat_dict[curr_t_str][0]=current_stat_dict[curr_t_str][0]+1
            elif match[7]=='Draw':
                current_stat_dict[curr_t_str][1]=current_stat_dict[curr_t_str][1]+1
            elif match[7]=='Lose':
                current_stat_dict[curr_t_str][2]=current_stat_dict[curr_t_str][2]+1
                
            wins=current_stat_dict[curr_t_str][0]
            wins_lst.append(wins) 
            draws=current_stat_dict[curr_t_str][1]
            draws_lst.append(draws)
            loses=current_stat_dict[curr_t_str][2]
            loses_lst.append(loses)
            
            
            current_stat_dict[curr_t_str][3]=current_stat_dict[curr_t_str][3]+match[9]
            scored=current_stat_dict[curr_t_str][3]
            scored_lst.append(scored)
            
            current_stat_dict[curr_t_str][4]=current_stat_dict[curr_t_str][4]+match[10]
            mised=current_stat_dict[curr_t_str][4]
            mised_lst.append(mised)
            
            # match result expectations
            expectation=elo_prob(current_rating_dict[match[4]], 
                                 current_rating_dict[match[5]]
                                )
            # new rating counting
            new_rating=elo_rating_changes(
                                    current_rating_dict[match[4]], 
                                    current_rating_dict[match[5]],
                                    score_dct[match[7]], match[12]
                                           )

            # rating changes    
            changing=new_rating-current_rating_dict[match[4]]

            # working dict update
            current_rating_dict.update({match[4]: new_rating})

            # opponent's rating changes
            opponent_rating=current_rating_dict[match[5]]-changing
        
        # else: no changings
        else:
            new_rating=current_rating_dict[match[4]]
            changing=0
            opponent_rating=current_rating_dict[match[5]]
            
            wins=current_stat_dict[curr_t_str][0]
            draws=current_stat_dict[curr_t_str][1]
            loses=current_stat_dict[curr_t_str][2]
            scored=current_stat_dict[curr_t_str][3]
            mised=current_stat_dict[curr_t_str][4]
            
            wins_lst.append(wins)
            draws_lst.append(draws)
            loses_lst.append(loses)
            scored_lst.append(scored)
            mised_lst.append(mised)


        # list for dataframe
        ratings_lst.append(new_rating)
        opponent_ratings_lst.append(opponent_rating)
        ratings_chng_lst.append(changing)
        
    # ratings AFTER match
    team_df['rating'] = ratings_lst
    team_df['opponent_rating'] = opponent_ratings_lst
    team_df['rating_changing'] = ratings_chng_lst   
    # personal teams stat AFTER match
    team_df['wins'] = wins_lst
    team_df['draws'] = draws_lst
    team_df['loses'] = loses_lst
    team_df['scored'] = scored_lst
    team_df['mised'] = mised_lst
    
    return team_df, current_rating_dict, current_stat_dict

# predicting game result by elo ratings
def elo_predict(team, opponent, draw_share):
    try:
        rating=current_rating_dict[team]
        opponent_rating=current_rating_dict[opponent]
        prob=elo_prob(rating, opponent_rating)
        draw_balance=0.5

        if prob>draw_balance+draw_share/2:
            return 'Win'
        elif prob<draw_balance-draw_share/2:
            return 'Lose'
        else:
            return 'Draw'
    except:
        return 'Unknown'
    
    
def elo_prob_predict(prob, draw_share):
    try:
        draw_balance=0.5

        if prob>draw_balance+draw_share/2:
            return 'Win'
        elif prob<draw_balance-draw_share/2:
            return 'Lose'
        else:
            return 'Draw'
    except:
        return 'Unknown'
# competitor function
#  home team always win
def simple_predict(team, opponent):
    return 'Win'

# competitor function
# random 
def random_predict(team, opponent, seed):
    np.random.seed(seed)
    return np.random.choice(['Win', 'Lose', 'Draw'])

# deprecated format functions
def get_season(lst):
    if len(lst)==5:
        return lst[3]
    else:
        return lst[2]
def get_division(txt):
    l=txt.split('.')
    return l[1]
def get_competition(txt):
    l=txt.split('.')
    return l[0]

# insert data in sql table
def insert_df(df, table_name):
    import datetime 
    now=datetime.datetime.now() 
    df['inserted_at']=now
    
    postgresql_engine=get_engine()
    
    df.to_sql(table_name, con=postgresql_engine, if_exists='append')   
    print('Ready: ', len(df), ' rows inserted')   
    
def get_feature_importance(clf, col):
    # feature importances
    importances = clf.feature_importances_

#     col=X_train.columns
    fi=pd.DataFrame(col, columns=['feature'])
    fi['importance']=importances
    fi=fi.sort_values(by='importance', ascending=False)
    fi.index=range(len(fi))
    fi['position']=fi.index
    # fi['importance'] = (fi['importance']*100).map('{:.2f}%'.format) 
    return fi

def bets_row(y, predict, bets):
    if y != predict:
        return 0
    else:
        if y == 'Win':
            cf = bets[0]
        elif y == 'Draw':
            cf = bets[1]
        elif y == 'Lose':
            cf = bets[2]
        return cf * 1

# all usefull metrics after working of algorithm
def count_metric(test_df, predict):
    import sklearn.metrics as skm
    
    try:
        test_df['revenue'] = test_df[['y', predict, 'cf1', 'cfdr', 'cf2']].apply(
                                            lambda x: bets_row(x[0], x[1], [
                                                                            x[2], x[3], x[4]
                                                                           ]
                                                              ), axis=1
                                                        )
    except:
        test_df['revenue'] = 0
    
    m=[
        len(test_df),
        skm.accuracy_score(test_df['y'], test_df[predict]),
#         skm.precision_score(test_df['y'], test_df[predict], average='macro'),
#         skm.recall_score(test_df['y'], test_df[predict], average='macro'),
        skm.f1_score(test_df['y'], test_df[predict], average='macro'),
        (np.sum(test_df['revenue']) / test_df['revenue'].count()) - 1,
    ]
    
    return m

# count_metric( with group by
def metric_by_segmens(test_df, predict, segment):
    segments=[]
    df_lst=[]
    for el in test_df[segment].value_counts().index:
        b=test_df[test_df[segment]==el]
        df_lst.append(b)
        segments.append(el)
    metrics=[]
    for el in df_lst:
        m=count_metric(el, predict)
        metrics.append(m)

    result=pd.DataFrame(metrics)
    result.columns = [
        'size', 
        'accuracy', 
#         'precision', 
#         'recall', 
        'f1',
        'roi',
    ]
    result['segment']=segments
    
    return result.sort_values(by=['size', 'f1'], ascending=[False, False])


def model_results(test_df, model_name, predict, predict_parameters):
    params = count_metric(test_df, predict)
    params.append(model_name)
    predict_parameters.append(params)
    
    test_df['global_segment'] = 1
    
    model_comparison_df = metric_by_segmens(test_df, predict, 'global_segment')
    model_comparison_df['model'] = model_name
    

    bins = test_df.groupby(['y', predict]).count()[['id']].reset_index()
    bins['y'] = bins['y'].map(score_label_dct)
    bins[predict] = bins[predict].map(score_label_dct)
    bins = bins.sort_values(by=['y', predict])
    bins['total'] = bins.groupby('y').id.transform(np.sum)
    bins['share'] = bins['id'] / bins['total']

    pv = bins.pivot_table(index='y', columns=predict, values='share').fillna(0)

    sns.heatmap(
                pv, 
                cmap="RdYlGn", 
                annot=True,
                fmt=".2f",
                center=0.50,
                )
    
    return model_comparison_df

# some graphs to look at prediction
def predict_shows(test_df):
    ax = metric_by_segmens(test_df, 'competition')[0:10].set_index('segment')[['f1']].plot(kind='barh', 
                                                                               title='Difference F1 of largest leagues')
    ax.invert_yaxis()

    ax = metric_by_segmens(test_df[test_df['competition']=='england'], 
                      'team')[0:10].set_index('segment')[['f1']].plot(kind='barh', 
                                                                      title='English best teams for predicting')
    ax.invert_yaxis()

    ax = metric_by_segmens(test_df[test_df['competition']=='spain'], 
                      'team')[0:10].set_index('segment')[['f1']].plot(kind='barh', 
                                                                      title='Spain best teams for predicting')
    ax.invert_yaxis()

    metric_by_segmens(test_df, 'date').set_index('segment')[['f1']].plot(title='F1 dynamics')
    
    
    return count_metric(test_df)

# update fileds in test dataframe
def get_test_fields(test_df, current_rating_dict, current_stat_dict):
    import datetime 
    test_df['propability']=test_df[['team', 'opponent']].apply(
                        lambda x: elo_prob(
                                        current_rating_dict.get(x[0], 2400), 
                                        current_rating_dict.get(x[1], 2400), 
                                        ), axis=1)


    test_df['wins']=test_df[['team', 'opponent']].apply(lambda x:
                                    current_stat_dict.get(x[0]+'-'+x[1], [0,0,0,0,0])[0],
                                                    axis=1)
    test_df['draws']=test_df[['team', 'opponent']].apply(lambda x:
                                    current_stat_dict.get(x[0]+'-'+x[1], [0,0,0,0,0])[1],
                                                    axis=1)
    test_df['loses']=test_df[['team', 'opponent']].apply(lambda x:
                                    current_stat_dict.get(x[0]+'-'+x[1], [0,0,0,0,0])[2],
                                                    axis=1)
    test_df['scored']=test_df[['team', 'opponent']].apply(lambda x:
                                    current_stat_dict.get(x[0]+'-'+x[1], [0,0,0,0,0])[3],
                                                    axis=1)
    test_df['mised']=test_df[['team', 'opponent']].apply(lambda x:
                                    current_stat_dict.get(x[0]+'-'+x[1], [0,0,0,0,0])[4],
                                                    axis=1)
    test_df['results_difference']=test_df['wins']-test_df['loses']
    test_df['goals_difference']=test_df['scored']-test_df['mised']


    test_df['propability_group']=np.floor(test_df['propability']*10)/10
    test_df['days_from_start']=(test_df['date']-datetime.datetime.strptime("2018-06-01", "%Y-%m-%d")).dt.days
    
    return test_df

def df_date_split(df, train_border1, train_border2, test_border1, test_border2):
    train_df = df[
        (df['date'] >= train_border1) &
        (df['date'] < train_border2)
                    ]

    test_df = df[
        (df['date'] >= test_border1) &
        (df['date'] < test_border2)
                    ]
    
    test_df = test_df[[
    'id',
    'team', 'opponent', 'venue',
    'team_id', 'opponent_team_id',
    'y'
    ]] 

    
    print('train, len:', len(train_df))
    print('test, len:', len(test_df))
    
#     X_train = train_df[x_fields]
#     y_train = train_df['y'].values

#     X_test = test_df[x_fields]
#     y_test = test_df['y'].values
    
    return train_df, test_df#, X_train, y_train, X_test, y_test


def normalize_df(df, x_fields_minus_plus, x_fields_zero_one, x_stable):
    minus_plus_x = pd.DataFrame(
        MinMaxScaler(feature_range=(-1,1)).fit_transform(
                                                            df[x_fields_minus_plus]
                                                        ), columns=x_fields_minus_plus
                                )

    zero_one_x = pd.DataFrame(
        MinMaxScaler(feature_range=(0,1)).fit_transform(
                                                            df[x_fields_zero_one]
                                                        ), columns=x_fields_zero_one
                            )

    direct_x = df[x_stable]

    X = pd.concat([
        minus_plus_x, 
        zero_one_x, 
        direct_x
                    ], axis=1)
    
    return X



def watching(test_df, predict, clf, x_fields, predict_parameters):
    hl = metric_by_segmens(test_df, predict, 'is_high_league')
    
    leagues = metric_by_segmens(test_df, predict, 'league_name')
    leagues = leagues.sort_values(by='segment', ascending=False)
    leagues.set_index('segment')[['f1']].plot(kind='barh')
    
    venue = metric_by_segmens(test_df, predict, 'venue')
    
    try:
        bin_size = 100
        test_df['rating_difference_bins'] = np.round(test_df['rating_difference'] / bin_size, 0) * bin_size
        metric_by_segmens(
            test_df, predict, 'rating_difference_bins'
        ).sort_values(by='segment').set_index('segment')[['f1']].plot()
    except Exception as e:
        print(str(e))
    
    try:
        fi = get_feature_importance(clf, x_fields)
    except:
        fi = pd.DataFrame()
        
    comparison_df = pd.DataFrame(predict_parameters, columns=[
        'games', 'accuracy', 'f1', 'roi', 'model_name']).drop_duplicates()

    return hl, leagues, venue, fi, comparison_df



def show_fields(X_train, threshold):
    corr_matrix = []
    col1_lst = []
    col2_lst = []
    for i in X_train.columns:
        for j in X_train.columns:
            r = stats.pearsonr(
                X_train[i], X_train[j]
                                )[0]
            corr_matrix.append(r)
            col1_lst.append(i)
            col2_lst.append(j)
    corr_df = pd.DataFrame(corr_matrix, columns=['correlation'])
    corr_df['field1'] = col1_lst
    corr_df['field2'] = col2_lst

    corr_pv = corr_df.pivot_table(index='field1', columns='field2', values='correlation').fillna(0)#.reset_index()

    problem_df = corr_df[
        (
            (corr_df['correlation'] > threshold) |
            (corr_df['correlation'] < -1 * threshold)
        ) &
        (corr_df['field1'] != corr_df['field2'])
    ]

    sns.heatmap(corr_pv, annot=True, fmt=".2g", cmap="YlGnBu", center=0)
    
    return problem_df

def hypeparams_fit(
                    X_train, y_train, 
                    X_test, test_df,
                    n_range,
                    d_range,
                    ):
    finding_lst = []
    for d in d_range:
        print(d)
        for n in n_range:
            print(n)
            clf = RandomForestClassifier(
                            n_estimators=n,
                            max_depth=d,
                            random_state=22)
            clf.fit(X_train, y_train)
            test_df['fitting_hyp_predict']=clf.predict(X_test)
            params = count_metric(test_df, 'fitting_hyp_predict')
            params.append(n)
            params.append(d)
            finding_lst.append(params)
    finding_df = pd.DataFrame(finding_lst)
    finding_df.columns = ['size', 'accuracy', 'f1', 'roi', 'n', 'depth'] 
    
    best_n = finding_df[finding_df['f1']==np.max(finding_df['f1'])]['n'].values[0]
    best_depth = finding_df[finding_df['f1']==np.max(finding_df['f1'])]['depth'].values[0]

    print('best_n', best_n)
    print('best_depth', best_depth)

    for_f1_hm = finding_df.pivot_table(
        index='n', 
        columns='depth', 
        values='f1').fillna(0)
    sns.heatmap(for_f1_hm, annot=True, fmt=".4g", cmap="YlGnBu")
    finding_df[finding_df['depth'] == best_depth].set_index('n')[['f1']].plot()
    finding_df[finding_df['n'] == best_n].set_index('depth')[['f1']].plot()
    
    return best_n, best_depth

# # используем .dot формат для визуализации дерева
# from sklearn.tree import export_graphviz
# export_graphviz(elo_clf.estimators_[0], feature_names=x_fields, 
# out_file='f_elo.dot', filled=True)
# # для этого понадобится библиотека pydot (pip install pydot)
# !dot -T svg 'f_elo.dot' -o 'f_elo.svg'