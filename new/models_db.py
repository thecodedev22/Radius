from sqlalchemy import Column, String, Float, DateTime, Text
from db_config import Base
import uuid

class EventDB(Base):
    __tablename__ = "events"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, index=True)
    description = Column(Text)
    url = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    source = Column(String)
