import pandas as pd
import numpy as np

def is_cloudy():
    """cloudy is an independent random variable.
    """
    return (np.random.random() < .5)
    
def is_sprinkler(cloudy):
    """sprinkler is dependent on cloudiness. if it is 
    cloudy, sprinkler has a 10% of being on. otherwise, 
    if it is sunny, then the sprinkler has a 50% of being 
    on.
    """
    if cloudy:
        return (np.random.random() < .1)
    else:
        return (np.random.random() < .5)

def is_rain(cloudy):
    if cloudy:
        return (np.random.random() < .8)
    else:
        return (np.random.random() < .2)
    
def is_wet(sprinkler, rain):
    if ((sprinkler == True) & (rain == True)):
        return (np.random.random() < .99)
    elif ((sprinkler == False) & (rain == True)):
        return (np.random.random() < .9)
    elif ((sprinkler == True) & (rain == False)):
        return (np.random.random() < .9)
    else:
        return False

# generate toy dataset via sampling (we couldve done this by multiplying probabilities)
n_samples = 10000

ls_c = []
ls_s = []
ls_r = []
ls_w = []

for _ in range(n_samples):
    c = is_cloudy()
    s = is_sprinkler(c)
    r = is_rain(c)
    w = is_wet(s, r)
    
    ls_c.append(c)
    ls_s.append(s)
    ls_r.append(r)
    ls_w.append(w)

collection = {"cloudy": ls_c, "sprinkler": ls_s, "rain": ls_r, "wet": ls_w}

# joint observations
df = pd.DataFrame(collection)
df.head()

df.to_csv("observations.csv")