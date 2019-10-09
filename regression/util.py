from env import host, user, password

def get_url(db):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

