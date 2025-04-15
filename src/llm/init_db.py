from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, Business
import uuid

# Database setup
DATABASE_URL = "sqlite:///./business_data.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize the database and create tables"""
    Base.metadata.create_all(bind=engine)

def create_test_business():
    """Create a test business if it doesn't exist"""
    db = SessionLocal()
    try:
        # Check if test business exists
        test_business = db.query(Business).filter(Business.name == "Test Business").first()
        if not test_business:
            test_business = Business(
                id=str(uuid.uuid4()),
                name="Test Business",
                description="A professional services company specializing in software development",
                business_hours="Monday-Friday 9AM-5PM PST",
                custom_instructions="""
                Please be professional and courteous. 
                Our main services include:
                - Custom software development
                - Cloud consulting
                - AI/ML solutions
                
                Common questions to handle:
                1. Pricing: Direct to sales team
                2. Technical support: Collect details and create ticket
                3. Job inquiries: Direct to careers page
                """,
                knowledge_base={
                    "services": [
                        "Custom software development",
                        "Cloud consulting",
                        "AI/ML solutions"
                    ],
                    "contact": {
                        "sales": "sales@testbusiness.com",
                        "support": "support@testbusiness.com"
                    },
                    "locations": ["San Francisco", "New York", "London"]
                }
            )
            db.add(test_business)
            db.commit()
            print(f"Created test business with ID: {test_business.id}")
        else:
            print("Test business already exists")
            
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Creating test business...")
    create_test_business()
    print("Database initialization complete") 