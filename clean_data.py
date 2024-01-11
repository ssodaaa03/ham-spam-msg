import pandas as pd

clean = lambda txt : ''.join( [ ch for ch in txt if ch.isalnum() or ch == ' ' ] )

file = 'SPAM text message 20170820 - Data.csv'

raw_data = pd.read_csv(file)

msg = raw_data['Message'].tolist()
cleaned_data = raw_data.replace(msg, list( map( clean, msg )))
cleaned_data = cleaned_data.dropna()

cleaned_data[:5000].to_csv('train_cleaned_'+file, encoding='utf-8', index=False)
cleaned_data[5000:].to_csv('test_cleaned_'+file, encoding='utf-8', index=False)

file_name = {'train' : 'train_cleaned_'+file,
             'test'  : 'test_cleaned_'+file  }

print('done...')