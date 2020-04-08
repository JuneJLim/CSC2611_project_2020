import re
import os
import pandas as pd
import pickle
from gensim.models import Word2Vec
import time

def tokenize_text(text: str) -> list:
    # pattern for tokens containing only alphabets
    pattern = re.compile("[a-zA-Z]+")
    
    t = text.lower()
    sents = re.split('; |\.|\?|\!', t) # list of sentences
    
    # for each sentence, generate a list of tokens
    # return a list of the lists of tokens
    return [ pattern.findall(s) for s in sents ]

def tokenize_file_party(path_to_file: str, party: list) -> list:
    
    # read in a .csv file
    df = pd.read_csv(path_to_file)
    
    text_list = []
        
    # if the proper name of party was passed
    if len(party) != 0:
        for row in df.itertuples():
            if str(row.speakerparty) in party:
                text_list = text_list + tokenize_text(str(row.speechtext))       
    # if the name of party was empty string
    else:
        # build a model with all the parties
        return tokenize_file(path_to_file)
        
    # return a list of the lists of tokens, per each file
    return text_list

def tokenize_file(path_to_file: str) -> list:
    
    # read in a .csv file
    df = pd.read_csv(path_to_file)
    
    text_list = []
    
    for text in df['speechtext']:
        text_list = text_list + tokenize_text(str(text))
        
    # return a list of the lists of tokens, per each file
    return text_list

def tokenize_month(path_to_month_folder: str, party: list) -> list:
    
    text_list = []
    
    for filename in os.listdir(path_to_month_folder):
        text_list = text_list + tokenize_file_party(path_to_month_folder + '\\' + filename, party)
        
    # return a list of the lists of tokens, per each folder
    # containing .csv files
    return text_list
    
def tokenize_year(path_to_year_folder: str, party: list) -> list:
    
    text_list = []
    
    for foldername in os.listdir(path_to_year_folder):
        text_list = text_list + tokenize_month(path_to_year_folder + '\\' + foldername, party)
        
    # return a list of the lists of tokens, per each year
    return text_list

def train_w2v(input_list: list, file_name: str):
    # set the size of window and dimension
    win = 20
    dim = 300 
    
    # train model (change the size of window)
    model = Word2Vec(input_list, window=win, size=dim, min_count=1)
    
    # summarize the loaded model
    #print(model)
    # summarize vocabulary
    #words = list(model.wv.vocab)
    #print(words)
    # access vector for one word
    #print(model['sentence'])

    # save model
    model.save("./"+str(dim)+"d_"+str(win)+"win"+file_name)
    # load model
    #new_model = Word2Vec.load(file_name)
    
    return model

def train_w2v_year_party(start_year: int, party: str):
    # how long is the time slice?
    years = 20 
    # last time slice lacks the data for 2020 thus one less year
    last_starting_year = 2001 

    # names of parties over time    
    party_lib = ['Liberal']
    party_con = ['Conservative', 'Conservative (1867-1942)', 'Progressive Conservative', 'Conservative Party of Canada', 'Liberal-Conservative', 'Canadian Alliance', 'Reform']
    party_dem = ['NDP', 'New Democratic Party', 'Co-operative Commonwealth Federation (C.C.F.)']
    
    # which party was passed?
    party_list = []
    if party == 'lib':
        party_list = party_lib
    elif party == 'con':
        party_list = party_con
    elif party == 'dem':
        party_list = party_dem
    elif party == 'libdem':
        party_list = party_dem + party_lib
    
    input_list = []
    end_year = 0
    
    # the last time slice will have one less year than the others
    if start_year == last_starting_year:
        end_year = 2019
    else:
        end_year = start_year + years
    
    # file name to be used when saving the model as a file
    file_name = "lipad" + str(start_year) + '-' + str(end_year) + party
    
    # iterate each directory for each year
    for i in range(start_year, end_year+1):
        print(i, party)
        path = '\lipad\\' + str(i)
        input_list = input_list + tokenize_year(os.getcwd() + path, party_list)
    
    # construct a model in the separate function    
    return train_w2v(input_list, file_name + '.bin')

def train_w2v_year_party_end_year(start_year: int, end_year: int, party: str):
    
    # names of parties over time    
    party_lib = ['Liberal']
    party_con = ['Conservative', 'Conservative (1867-1942)', 'Progressive Conservative', 'Conservative Party of Canada', 'Liberal-Conservative', 'Canadian Alliance', 'Reform']
    party_dem = ['NDP', 'New Democratic Party', 'Co-operative Commonwealth Federation (C.C.F.)']
    
    # which party was passed?
    party_list = []
    if party == 'lib':
        party_list = party_lib
    elif party == 'con':
        party_list = party_con
    elif party == 'dem':
        party_list = party_dem
    
    input_list = []
    
    # file name to be used when saving the model as a file
    file_name = "lipad" + str(start_year) + '-' + str(end_year) + party
    
    # iterate each directory for each year
    for i in range(start_year, end_year):
        print(i, party)
        path = '\lipad\\' + str(i)
        input_list = input_list + tokenize_year(os.getcwd() + path, party_list)
    
    # construct a model in the separate function    
    return train_w2v(input_list, file_name + '.bin')
    
if __name__ == '__main__':
    
    # one time slice = 20 years
    
    start = time.time()
    
    train_w2v_year_party(2001, '')
    train_w2v_year_party(1981, '')
    train_w2v_year_party(1961, '')
    train_w2v_year_party(1941, '')
    train_w2v_year_party(1921, '')
    train_w2v_year_party(1901, '')

    end = time.time()
    elapsed = end - start
    print(elapsed)
    
    # one time slice = 20 years
    # three parties separately except for the first time slice
    # (NDP/CCF did not exist)
    
    start = time.time()
    
    train_w2v_year_party(2001, 'lib')
    train_w2v_year_party(2001, 'con')
    train_w2v_year_party(2001, 'dem')
    
    end = time.time()
    elapsed = end - start
    print(elapsed)
    
    start = time.time()
    
    train_w2v_year_party(1981, 'lib')
    train_w2v_year_party(1981, 'con')
    train_w2v_year_party(1981, 'dem')
    
    end = time.time()
    elapsed = end - start
    print(elapsed)
    
    start = time.time()
    
    train_w2v_year_party(1961, 'lib')
    train_w2v_year_party(1961, 'con')
    train_w2v_year_party(1961, 'dem')
    
    end = time.time()
    elapsed = end - start
    print(elapsed)
    
    start = time.time()
    
    train_w2v_year_party(1941, 'lib')
    train_w2v_year_party(1941, 'con')
    train_w2v_year_party(1941, 'dem')
    
    end = time.time()
    elapsed = end - start
    print(elapsed)
    
    start = time.time()
    
    train_w2v_year_party(1921, 'lib')
    train_w2v_year_party(1921, 'con')
    train_w2v_year_party(1921, 'dem')
    
    end = time.time()
    elapsed = end - start
    print(elapsed)

    start = time.time()

    train_w2v_year_party(1901, 'lib')
    train_w2v_year_party(1901, 'con')
    # didn't exist before 1920
    #train_w2v_year_party(1901, 'dem') 

    end = time.time()
    elapsed = end - start
    print(elapsed)
    
    # one time slice = 5 years (from 1985-2019)
    # three parties separately
    
    year = 1986
    
    while year < 2017:
        start = time.time()
        
        train_w2v_year_party(year, 'libdem')
        
        end = time.time()
        elapsed = end - start
        print(year, "done", elapsed)
        
        year += 5