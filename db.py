import sqlalchemy
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer,String,DateTime,Float
from sqlalchemy.ext import declarative
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Prediction(Base):
    __tablename__ ='prediction'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    features =Column(String)
    result =Column(String)
    output =Column(String)
    created_on = Column(DateTime, default=datetime.now)

    def __str__(self):
        return self.name

if __name__ == "__main__":
    engine = create_engine('sqlite:///db.sqlite3')
    Base.metadata.create_all(engine)
    