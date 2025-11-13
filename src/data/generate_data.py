"""
============================================================================
BANKING TRANSACTION DATA GENERATOR
============================================================================

Hey there! Welcome to the data generator.

Think of this as a "transaction simulator" - we're creating fake but realistic
banking transactions that include both normal activity AND suspicious patterns
that money launderers might use.

WHY CREATE FAKE DATA?
----------------------
1. Real banking data is highly confidential (and illegal to share!)
2. We need labeled data (knowing which transactions are fraud)
3. We can control the patterns to test our models thoroughly
4. It's safe to experiment and learn without risk

WHAT THIS FILE DOES:
--------------------
1. Creates realistic customer profiles
2. Generates normal transactions (groceries, bills, etc.)
3. Injects AML fraud patterns (structuring, layering, etc.)
4. Adds text descriptions (mixed with numerical data)
5. Saves everything in a format PySpark can process

Let's dive in! 🚀
============================================================================
"""

# Import all the tools we need
# -----------------------------
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import os
import json

# Set random seed for reproducibility
# ------------------------------------
# This ensures we get the same "random" data every time we run the script
# Think of it like setting a starting point for randomness
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class BankingDataGenerator:
    """
    This class generates realistic banking transaction data.

    Think of a class as a blueprint - like a recipe that contains both
    the ingredients (data) and instructions (methods) for making transactions.
    """

    def __init__(
        self,
        num_customers: int = 1000,
        num_transactions: int = 50000,
        fraud_ratio: float = 0.20  # 20% fraud transactions (realistic for training)
    ):
        """
        Initialize our data generator.

        Parameters:
        -----------
        num_customers : int
            How many fake customer accounts to create
            (Like creating 1000 fictional bank customers)

        num_transactions : int
            Total number of transactions to generate
            (50,000 transactions = enough data to train good models)

        fraud_ratio : float
            What percentage should be fraudulent
            (0.20 = 20% fraud - higher than reality but good for learning)
        """
        self.num_customers = num_customers
        self.num_transactions = num_transactions
        self.fraud_ratio = fraud_ratio

        # Calculate how many of each type
        self.num_fraud = int(num_transactions * fraud_ratio)
        self.num_legitimate = num_transactions - self.num_fraud

        print(f"""
        🏦 Banking Data Generator Initialized
        =====================================
        Total Customers: {num_customers:,}
        Total Transactions: {num_transactions:,}
        └─ Legitimate: {self.num_legitimate:,} ({(1-fraud_ratio)*100:.1f}%)
        └─ Fraudulent: {self.num_fraud:,} ({fraud_ratio*100:.1f}%)

        Ready to generate data! 🎲
        """)

        # Define transaction categories for our AML detection
        # ---------------------------------------------------
        # These are the 5 classes our model will learn to predict
        self.categories = [
            'LEGITIMATE',      # Normal, everyday transactions
            'STRUCTURING',     # Breaking large amounts into smaller ones
            'LAYERING',        # Complex chains to hide money origin
            'SHELL_COMPANY',   # Fake business transactions
            'ROUND_TRIPPING'   # Circular money movements
        ]

        # Lists of realistic transaction descriptions
        # -------------------------------------------
        # These help make our data feel real and provide text features

        self.legitimate_descriptions = [
            "Grocery store purchase", "Gas station payment", "Restaurant bill",
            "Online shopping", "Utility bill payment", "Rent payment",
            "Pharmacy purchase", "Coffee shop", "Movie tickets",
            "Gym membership", "Insurance payment", "Phone bill",
            "Internet service", "Streaming subscription", "Book purchase",
            "Clothing store", "Electronics purchase", "Home supplies",
            "Pet supplies", "Medical copay", "Parking fee",
            "Public transportation", "Haircut", "Car maintenance"
        ]

        self.structuring_descriptions = [
            "Cash deposit below threshold", "ATM withdrawal series",
            "Multiple small transfers", "Sequential deposits",
            "Cash deposit at branch", "Repeated wire transfers",
            "Small amount wire", "Below reporting limit",
            "Split transaction", "Fragmented deposit"
        ]

        self.layering_descriptions = [
            "International wire transfer", "Cross-border payment",
            "Multi-hop transfer", "Foreign exchange transaction",
            "Offshore account transfer", "Complex routing",
            "Chain transaction", "Intermediary payment",
            "Third-party transfer", "Nested transaction"
        ]

        self.shell_company_descriptions = [
            "Business consulting fee", "Management services",
            "Advisory payment", "Professional services",
            "Consulting agreement", "Service contract payment",
            "Business development fee", "Strategic consulting",
            "Corporate services", "Business advisory"
        ]

        self.round_tripping_descriptions = [
            "Return of investment", "Loan repayment",
            "Capital return", "Investment redemption",
            "Funds repatriation", "Circular payment",
            "Mirror transaction", "Reciprocal transfer",
            "Matching payment", "Symmetric transaction"
        ]

        # Realistic names for our fake customers
        # --------------------------------------
        self.first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
            "Michael", "Linda", "William", "Elizabeth", "David", "Barbara",
            "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
            "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
            "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"
        ]

        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
            "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
            "Jackson", "Martin", "Lee", "Thompson", "White", "Harris",
            "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker"
        ]

    def generate_customer_profiles(self) -> pd.DataFrame:
        """
        Create realistic customer profiles.

        Think of this as creating fake ID cards for our customers,
        complete with account numbers, names, and risk scores.

        Returns:
        --------
        DataFrame with customer information
        """
        print("👤 Creating customer profiles...")

        customers = []

        for i in range(self.num_customers):
            # Generate a unique customer ID (like an account number)
            customer_id = f"CUST_{i:06d}"

            # Create a random name
            name = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"

            # Account age in days (older accounts are usually less risky)
            account_age_days = random.randint(30, 3650)  # 1 month to 10 years

            # Average account balance (ranges from broke to wealthy)
            avg_balance = random.choice([
                random.uniform(100, 5000),      # 60% low balance
                random.uniform(5000, 50000),    # 30% medium balance
                random.uniform(50000, 500000)   # 10% high balance
            ])

            # Historical risk score (0-1, where 1 is riskiest)
            # Most customers are low risk, few are high risk
            risk_score = np.random.beta(2, 5)  # Beta distribution skewed toward low values

            customers.append({
                'customer_id': customer_id,
                'name': name,
                'account_age_days': account_age_days,
                'avg_balance': avg_balance,
                'risk_score': risk_score
            })

        df = pd.DataFrame(customers)
        print(f"✅ Created {len(df)} customer profiles")
        return df

    def generate_legitimate_transaction(
        self,
        transaction_id: int,
        customer_id: str,
        timestamp: datetime
    ) -> Dict:
        """
        Generate a normal, everyday transaction.

        These are the boring but important transactions:
        buying groceries, paying bills, etc.

        Parameters:
        -----------
        transaction_id : int
            Unique ID for this transaction
        customer_id : str
            Which customer is making this transaction
        timestamp : datetime
            When the transaction happens

        Returns:
        --------
        Dictionary with transaction details
        """
        # Legitimate transactions are usually small to medium amounts
        # Most people spend $10-$500 on daily transactions
        amount = random.choice([
            random.uniform(5, 100),      # 70% small purchases
            random.uniform(100, 500),    # 25% medium purchases
            random.uniform(500, 5000)    # 5% large purchases (rent, etc.)
        ])

        # Pick a random description
        description = random.choice(self.legitimate_descriptions)

        # Add more realistic details
        merchant_category = random.choice([
            'RETAIL', 'FOOD', 'UTILITIES', 'ENTERTAINMENT',
            'HEALTHCARE', 'TRANSPORTATION', 'SERVICES'
        ])

        # Most legitimate transactions are domestic
        location = random.choice(['US'] * 95 + ['CA', 'MX', 'UK', 'EU'])

        return {
            'transaction_id': f"TXN_{transaction_id:08d}",
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'description': description,
            'category': 'LEGITIMATE',
            'merchant_category': merchant_category,
            'location': location,
            'is_fraud': 0  # Not fraud
        }

    def generate_structuring_transaction(
        self,
        transaction_id: int,
        customer_id: str,
        timestamp: datetime
    ) -> Dict:
        """
        Generate a STRUCTURING transaction.

        WHAT IS STRUCTURING?
        --------------------
        Criminals break large amounts into smaller transactions to avoid
        triggering reporting requirements. For example:
        - Instead of depositing $50,000 (which gets reported)
        - They deposit $9,000 ten times over several days

        Key Pattern: Multiple transactions just below $10,000 threshold
        """
        # Structuring amounts are typically just below reporting thresholds
        # In the US, banks report transactions over $10,000
        amount = random.uniform(7000, 9900)  # Just under $10k

        description = random.choice(self.structuring_descriptions)

        return {
            'transaction_id': f"TXN_{transaction_id:08d}",
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'description': description,
            'category': 'STRUCTURING',
            'merchant_category': 'CASH_DEPOSIT',
            'location': 'US',
            'is_fraud': 1  # This is fraud!
        }

    def generate_layering_transaction(
        self,
        transaction_id: int,
        customer_id: str,
        timestamp: datetime
    ) -> Dict:
        """
        Generate a LAYERING transaction.

        WHAT IS LAYERING?
        -----------------
        Moving money through multiple accounts and jurisdictions to
        obscure its origin. Like a shell game with money!

        Example:
        Criminal's money → Offshore account → Different country →
        Another account → Back to criminal (now looks "clean")

        Key Pattern: International transfers, complex routing
        """
        # Layering involves larger amounts moving internationally
        amount = random.uniform(10000, 500000)

        description = random.choice(self.layering_descriptions)

        # Usually involves international locations
        location = random.choice([
            'CAYMAN_ISLANDS', 'SWITZERLAND', 'PANAMA', 'BAHAMAS',
            'LUXEMBOURG', 'HONG_KONG', 'SINGAPORE', 'DUBAI'
        ])

        return {
            'transaction_id': f"TXN_{transaction_id:08d}",
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'description': description,
            'category': 'LAYERING',
            'merchant_category': 'INTERNATIONAL_WIRE',
            'location': location,
            'is_fraud': 1
        }

    def generate_shell_company_transaction(
        self,
        transaction_id: int,
        customer_id: str,
        timestamp: datetime
    ) -> Dict:
        """
        Generate a SHELL COMPANY transaction.

        WHAT IS A SHELL COMPANY?
        ------------------------
        A fake business that exists only on paper, used to move
        illegal money while making it look like legitimate business.

        Example:
        Criminal creates "ABC Consulting LLC" (doesn't actually do anything)
        → Sends illegal money as "consulting fees"
        → Money now looks like legitimate business income

        Key Pattern: Vague business services, round amounts, regular payments
        """
        # Shell company transactions often have suspiciously round amounts
        amount = random.choice([
            10000, 15000, 25000, 50000, 75000, 100000
        ]) + random.uniform(0, 1000)  # Add small variation

        description = random.choice(self.shell_company_descriptions)

        return {
            'transaction_id': f"TXN_{transaction_id:08d}",
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'description': description,
            'category': 'SHELL_COMPANY',
            'merchant_category': 'BUSINESS_SERVICES',
            'location': random.choice(['US', 'DELAWARE', 'NEVADA', 'WYOMING']),
            'is_fraud': 1
        }

    def generate_round_tripping_transaction(
        self,
        transaction_id: int,
        customer_id: str,
        timestamp: datetime
    ) -> Dict:
        """
        Generate a ROUND TRIPPING transaction.

        WHAT IS ROUND TRIPPING?
        -----------------------
        Money goes out and comes back in through complex paths,
        making it appear as legitimate investment or loan activity.

        Example:
        Criminal sends $100k to Company A →
        Company A sends to Company B →
        Company B sends $95k back to criminal as "investment return"
        → Original dirty money now looks like legitimate profit

        Key Pattern: Symmetrical transactions, matching amounts
        """
        # Round tripping often involves matching amounts
        amount = random.uniform(20000, 1000000)

        description = random.choice(self.round_tripping_descriptions)

        return {
            'transaction_id': f"TXN_{transaction_id:08d}",
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'description': description,
            'category': 'ROUND_TRIPPING',
            'merchant_category': 'INVESTMENT',
            'location': random.choice(['US', 'OFFSHORE']),
            'is_fraud': 1
        }

    def generate_all_transactions(
        self,
        customers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate all transactions (both legitimate and fraudulent).

        This is the main method that creates our complete dataset
        by mixing normal and suspicious transactions.

        Parameters:
        -----------
        customers_df : DataFrame
            Our customer profiles

        Returns:
        --------
        DataFrame with all transactions
        """
        print("\n💳 Generating transactions...")
        print(f"├─ Legitimate: {self.num_legitimate:,}")
        print(f"└─ Fraudulent: {self.num_fraud:,}")

        transactions = []
        customer_ids = customers_df['customer_id'].tolist()

        # Start date: 1 year ago from now
        start_date = datetime.now() - timedelta(days=365)

        # Generate legitimate transactions
        # --------------------------------
        print("\n🛒 Creating legitimate transactions...")
        for i in range(self.num_legitimate):
            # Random customer
            customer_id = random.choice(customer_ids)

            # Random timestamp within the past year
            random_days = random.randint(0, 365)
            random_seconds = random.randint(0, 86400)  # seconds in a day
            timestamp = start_date + timedelta(days=random_days, seconds=random_seconds)

            txn = self.generate_legitimate_transaction(i, customer_id, timestamp)
            transactions.append(txn)

            # Progress indicator
            if (i + 1) % 10000 == 0:
                print(f"  ├─ Generated {i+1:,} legitimate transactions")

        print(f"  ✅ Completed {self.num_legitimate:,} legitimate transactions")

        # Generate fraudulent transactions
        # --------------------------------
        print("\n🚨 Creating fraudulent transactions...")

        # Distribute fraud types evenly
        fraud_types = ['structuring', 'layering', 'shell_company', 'round_tripping']
        fraud_per_type = self.num_fraud // len(fraud_types)

        fraud_generators = {
            'structuring': self.generate_structuring_transaction,
            'layering': self.generate_layering_transaction,
            'shell_company': self.generate_shell_company_transaction,
            'round_tripping': self.generate_round_tripping_transaction
        }

        fraud_idx = self.num_legitimate

        for fraud_type in fraud_types:
            print(f"\n  📍 Generating {fraud_type.upper()} transactions...")
            generator = fraud_generators[fraud_type]

            for i in range(fraud_per_type):
                customer_id = random.choice(customer_ids)
                random_days = random.randint(0, 365)
                random_seconds = random.randint(0, 86400)
                timestamp = start_date + timedelta(days=random_days, seconds=random_seconds)

                txn = generator(fraud_idx, customer_id, timestamp)
                transactions.append(txn)
                fraud_idx += 1

                if (i + 1) % 2000 == 0:
                    print(f"    ├─ Generated {i+1:,} {fraud_type} transactions")

            print(f"    ✅ Completed {fraud_per_type:,} {fraud_type} transactions")

        # Create DataFrame
        df = pd.DataFrame(transactions)

        # Sort by timestamp (chronological order - more realistic)
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"\n✅ Total transactions generated: {len(df):,}")
        print(f"\nClass distribution:")
        print(df['category'].value_counts().to_string())

        return df

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated features that might help detect fraud.

        FEATURE ENGINEERING is where the magic happens!
        Instead of just using raw data, we create new features
        that make patterns more obvious to the model.

        Think of it like this:
        - Raw feature: "amount = $9,500"
        - Engineered feature: "is_near_threshold = True" (more informative!)

        Parameters:
        -----------
        df : DataFrame
            Our transaction data

        Returns:
        --------
        DataFrame with additional calculated features
        """
        print("\n🔧 Adding engineered features...")

        # Time-based features
        # -------------------
        # Fraud patterns often vary by time
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].between(22, 6).astype(int)

        # Amount-based features
        # ---------------------
        # Red flags related to transaction amounts
        df['is_round_amount'] = (df['amount'] % 1000 < 10).astype(int)
        df['is_large_amount'] = (df['amount'] > 10000).astype(int)
        df['is_near_threshold'] = ((df['amount'] >= 7000) & (df['amount'] < 10000)).astype(int)
        df['amount_log'] = np.log1p(df['amount'])  # Log transform helps with skewed data

        # Text-based features
        # -------------------
        # Simple text features (more advanced NLP will come later)
        df['description_length'] = df['description'].str.len()
        df['description_word_count'] = df['description'].str.split().str.len()

        # International transaction flag
        df['is_international'] = (~df['location'].isin(['US'])).astype(int)

        print("✅ Added engineered features:")
        print("  ├─ Time features: hour, day_of_week, is_weekend, is_night")
        print("  ├─ Amount features: is_round_amount, is_large_amount, is_near_threshold")
        print("  ├─ Text features: description_length, description_word_count")
        print("  └─ Location features: is_international")

        return df

    def save_data(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        output_dir: str = "data/raw"
    ):
        """
        Save our generated data to disk.

        We save in multiple formats:
        - CSV: Easy to open in Excel, human-readable
        - Parquet: Efficient format for PySpark
        - JSON: Metadata and statistics

        Parameters:
        -----------
        customers_df : DataFrame
            Customer profiles
        transactions_df : DataFrame
            Transaction data
        output_dir : str
            Where to save the files
        """
        print(f"\n💾 Saving data to {output_dir}...")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save customers
        customers_file = os.path.join(output_dir, "customers.csv")
        customers_df.to_csv(customers_file, index=False)
        print(f"  ✅ Saved: {customers_file}")

        # Save transactions (CSV for exploration)
        transactions_csv = os.path.join(output_dir, "transactions.csv")
        transactions_df.to_csv(transactions_csv, index=False)
        print(f"  ✅ Saved: {transactions_csv}")

        # Save transactions (Parquet for PySpark - much faster!)
        transactions_parquet = os.path.join(output_dir, "transactions.parquet")
        transactions_df.to_parquet(transactions_parquet, index=False)
        print(f"  ✅ Saved: {transactions_parquet}")

        # Save metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'num_customers': len(customers_df),
            'num_transactions': len(transactions_df),
            'fraud_ratio': self.fraud_ratio,
            'date_range': {
                'start': transactions_df['timestamp'].min().isoformat(),
                'end': transactions_df['timestamp'].max().isoformat()
            },
            'class_distribution': transactions_df['category'].value_counts().to_dict(),
            'columns': list(transactions_df.columns)
        }

        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Saved: {metadata_file}")

        # Print summary statistics
        print("\n📊 Data Summary")
        print("="*50)
        print(f"Total Customers: {len(customers_df):,}")
        print(f"Total Transactions: {len(transactions_df):,}")
        print(f"\nAmount Statistics:")
        print(f"  ├─ Min: ${transactions_df['amount'].min():,.2f}")
        print(f"  ├─ Max: ${transactions_df['amount'].max():,.2f}")
        print(f"  ├─ Mean: ${transactions_df['amount'].mean():,.2f}")
        print(f"  └─ Median: ${transactions_df['amount'].median():,.2f}")
        print(f"\nClass Balance:")
        for category, count in transactions_df['category'].value_counts().items():
            percentage = (count / len(transactions_df)) * 100
            print(f"  ├─ {category}: {count:,} ({percentage:.1f}%)")

        print("\n🎉 Data generation complete!")


def main():
    """
    Main function to run the data generation pipeline.

    This is what gets executed when you run:
        python src/data/generate_data.py
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🏦  BANKING AML TRANSACTION DATA GENERATOR  🏦             ║
    ║                                                              ║
    ║   Creating realistic banking data for fraud detection       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    # -------------
    # Feel free to adjust these numbers!
    NUM_CUSTOMERS = 1000      # Number of fake customers
    NUM_TRANSACTIONS = 50000  # Total transactions
    FRAUD_RATIO = 0.20        # 20% will be fraudulent

    # Initialize generator
    generator = BankingDataGenerator(
        num_customers=NUM_CUSTOMERS,
        num_transactions=NUM_TRANSACTIONS,
        fraud_ratio=FRAUD_RATIO
    )

    # Step 1: Create customer profiles
    customers_df = generator.generate_customer_profiles()

    # Step 2: Generate all transactions
    transactions_df = generator.generate_all_transactions(customers_df)

    # Step 3: Add engineered features
    transactions_df = generator.add_engineered_features(transactions_df)

    # Step 4: Save everything
    generator.save_data(customers_df, transactions_df)

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                     ✅ SUCCESS! ✅                           ║
    ║                                                              ║
    ║  Your banking transaction dataset is ready!                 ║
    ║                                                              ║
    ║  Next steps:                                                ║
    ║  1. Explore the data: open data/raw/transactions.csv        ║
    ║  2. Check metadata: data/raw/metadata.json                  ║
    ║  3. Run the ML pipeline: python src/main_pipeline.py        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    # This runs only when you execute this file directly
    main()
