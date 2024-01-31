import pandas as pd
import numpy as np

# def get_ngrams(TOKEN, n=2, sent_key='sent_num'):
    
#     OHCO = TOKEN.index.names
#     grouper = list(OHCO)[:OHCO.index(sent_key)+1]

#     PADDED = TOKEN.groupby(grouper)\
#         .apply(lambda x: '<s> ' + ' '.join(x.term_str) + ' </s>')\
#         .apply(lambda x: pd.Series(x.split()))\
#         .stack().to_frame('term_str')
#     PADDED.index.names = grouper + ['token_num']
        
#     for i in range(1, n):
#         PADDED = PADDED.join(PADDED.term_str.shift(-i), rsuffix=i)

#     PADDED.columns = [f'w{j}' for j in range(n)]

#     PADDED = PADDED.fillna('<s>')
#     # PADDED = PADDED[~((PADDED.w0 == '</s>') & (PADDED[f'w{n-1}'] == '<s>'))]

#     return PADDED


def get_ngrams(TOKEN, n=2, sent_key='sent_num'):
    
    OHCO = TOKEN.index.names
    grouper = list(OHCO)[:OHCO.index(sent_key)+1]

    PADDED = TOKEN.groupby(grouper)\
        .apply(lambda x: '<s> ' + ' '.join(x.term_str) + ' </s>')\
        .apply(lambda x: pd.Series(x.split()))\
        .stack().to_frame('term_str')
    PADDED.index.names = grouper + ['token_num']

    NGRAMS = PADDED.groupby(grouper)\
        .apply(lambda x: pd.concat([x.shift(0-i) for i in range(n)], axis=1)).reset_index(drop=True)
    NGRAMS.index = PADDED.index
    NGRAMS.columns = [f'w{j}' for j in range(n)]
    
    return NGRAMS


def get_ngram_counts(NGRAM):
    "Compress the sequences into counts"
    
    n = len(NGRAM.columns)
    C = [None for i in range(n)]
    
    for i in range(n):

        # Count distinct ngrams
        C[i] = NGRAM.iloc[:, :i+1].value_counts().to_frame('n').sort_index()
    
        # Get joint probabilities (MLE)
        C[i]['p'] = C[i].n / C[i].n.sum()
        C[i]['i'] = np.log2(1/C[i].p)

        # Get conditional probabilities (MLE)
        if i > 0:
            C[i]['cp'] = C[i].n / C[i-1].n
            C[i]['ci'] = np.log2(1/C[i].cp)
            
    
    # Convert index vales to scalars from single tuples        
    C[0].index = [i[0] for i in C[0].index]
    C[0].index.name = 'w0'
            
    return C

def test_model(model, test_ngrams):

    # Get the model level and info feature
    n = len(model.index.names) - 1 
    f = 'c' * bool(n) + 'i'        

    # Do the test by join and then split-apply-combine
    # fillna() is used to hanlde OOV terms
    T = test_ngrams.join(model[f], on=model.index.names, how='left').fillna(model[f].max()) #.copy()
        
    R = T.groupby('sent_num')[f].agg(['sum','mean'])
    R['pp'] = np.exp2(R['mean'])
    
    return R

def generate_text(M, n=250):
    
    if len(M) < 3:
        raise ValueError("Must have trigram model generated.")
    
    # Start list of words
    words = ['</s>', '<s>']
    
    for i in range(n):
        
        bg = tuple(words[-2:])

        # Try trigram model
        try:
            next_word = M[2].loc[bg].sample(weights='cp').index[0]

        # If not found in model, back off ...
        except KeyError:
            
            print("BACKOFF!")

            # Get the last word in the bigram
            ug = bg[1]
            next_word = M[1].loc[ug].sample(weights='cp').index[0]
                    
        words.append(next_word)
    
    text = ' '.join(words[2:])
    print(text.replace(' </s> <s> ', '.\n\n').upper()+".")