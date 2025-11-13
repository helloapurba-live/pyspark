"""
=============================================================================
BANKING TRANSACTION DATA GENERATOR
=============================================================================
This module creates realistic synthetic banking transaction data for AML fraud detection.

Think of this as creating a "practice dataset" - it looks like real banking data
but it's completely fake and safe to use for learning!

We'll generate:
1. TABULAR DATA: Numbers like transaction amounts, account age, etc.
2. TEXT DATA: Transaction descriptions, merchant categories, customer notes
3. FRAUD LABELS: Which transactions are fraudulent (what we want to predict)
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
from typing import Tuple, List
import logging

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankingDataGenerator:
    """
    This class generates synthetic banking transaction data.

    Why use a class? It keeps all our data generation logic organized
    in one place, like keeping all your cooking utensils in one drawer!
    """

    def __init__(self, num_transactions: int = 100000, fraud_ratio: float = 0.02,
                 random_seed: int = 42):
        """
        Initialize the data generator.

        Parameters:
        -----------
        num_transactions : int
            How many transactions to generate (default 100,000)
        fraud_ratio : float
            Percentage of fraudulent transactions (default 2%)
        random_seed : int
            Random seed for reproducibility (same seed = same data every time)
        """
        self.num_transactions = num_transactions
        self.fraud_ratio = fraud_ratio
        self.random_seed = random_seed

        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Initialize Faker for generating realistic fake data
        self.fake = Faker()
        Faker.seed(random_seed)

        logger.info(f"📊 Initialized data generator for {num_transactions} transactions")
        logger.info(f"🚨 Fraud ratio: {fraud_ratio*100}%")

    def generate_data(self) -> pd.DataFrame:
        """
        Generate complete banking transaction dataset.

        This is the main method that creates all our data!

        Returns:
        --------
        pd.DataFrame : Complete dataset with all features and labels
        """
        logger.info("🏗️  Starting data generation...")

        # Calculate how many fraudulent transactions we need
        num_fraud = int(self.num_transactions * self.fraud_ratio)
        num_legitimate = self.num_transactions - num_fraud

        logger.info(f"   ✓ Legitimate transactions: {num_legitimate}")
        logger.info(f"   ✗ Fraudulent transactions: {num_fraud}")

        # Generate legitimate and fraudulent transactions separately
        # Why? Because fraud has different patterns than legitimate transactions!
        legitimate_data = self._generate_legitimate_transactions(num_legitimate)
        fraudulent_data = self._generate_fraudulent_transactions(num_fraud)

        # Combine both types of transactions
        data = pd.concat([legitimate_data, fraudulent_data], ignore_index=True)

        # Shuffle the data so fraud isn't all at the end
        data = data.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        # Add transaction IDs
        data['transaction_id'] = [f"TXN{str(i).zfill(8)}" for i in range(len(data))]

        logger.info(f"✅ Generated {len(data)} total transactions")
        logger.info(f"📋 Features: {data.columns.tolist()}")

        return data

    def _generate_legitimate_transactions(self, num_samples: int) -> pd.DataFrame:
        """
        Generate legitimate (non-fraudulent) transactions.

        These follow normal banking patterns:
        - Reasonable amounts
        - Regular timing
        - Familiar merchants
        - Normal descriptions
        """
        logger.info("💳 Generating legitimate transactions...")

        data = {
            # === TABULAR FEATURES (Numbers and Categories) ===

            # Transaction amount (most legitimate transactions are small)
            # Using log-normal distribution - most are small, few are large
            'transaction_amount': np.random.lognormal(mean=4.5, sigma=1.2, size=num_samples),

            # Account age in days (older accounts are typically more trustworthy)
            'account_age_days': np.random.gamma(shape=2, scale=365, size=num_samples),

            # Number of transactions in last 24 hours (legitimate users don't transact too much)
            'num_transactions_24h': np.random.poisson(lam=2, size=num_samples),

            # Average transaction amount for this account
            'avg_transaction_amount': np.random.lognormal(mean=4.0, sigma=1.0, size=num_samples),

            # Transaction hour (0-23) - legitimate transactions happen during business hours
            'transaction_hour': np.random.choice(range(8, 22), size=num_samples,
                                                p=[0.1]*14/sum([0.1]*14)),

            # Day of week (0=Monday, 6=Sunday) - more transactions on weekdays
            'transaction_day_of_week': np.random.choice(range(7), size=num_samples,
                                                       p=[0.18, 0.18, 0.18, 0.18, 0.18, 0.05, 0.05]),

            # Customer age (legitimate banking customers)
            'customer_age': np.random.normal(loc=45, scale=15, size=num_samples).astype(int),

            # Credit score (higher is better)
            'credit_score': np.random.normal(loc=700, scale=80, size=num_samples).astype(int),

            # Number of failed login attempts (legitimate users rarely fail)
            'num_failed_logins': np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05]),

            # Label: 0 = legitimate
            'is_fraud': np.zeros(num_samples, dtype=int)
        }

        # Clip values to realistic ranges
        data['transaction_amount'] = np.clip(data['transaction_amount'], 1, 50000)
        data['account_age_days'] = np.clip(data['account_age_days'], 1, 7300)  # Max 20 years
        data['customer_age'] = np.clip(data['customer_age'], 18, 90)
        data['credit_score'] = np.clip(data['credit_score'], 300, 850)

        df = pd.DataFrame(data)

        # === CATEGORICAL FEATURES ===
        df['source_country'] = [self._get_legitimate_country() for _ in range(num_samples)]
        df['destination_country'] = [self._get_legitimate_country() for _ in range(num_samples)]
        df['account_type'] = np.random.choice(['checking', 'savings', 'credit'],
                                             size=num_samples, p=[0.6, 0.3, 0.1])

        # === TEXT FEATURES ===
        df['transaction_description'] = [self._get_legitimate_description()
                                        for _ in range(num_samples)]
        df['merchant_category'] = [self._get_legitimate_merchant()
                                  for _ in range(num_samples)]
        df['customer_notes'] = [self._get_legitimate_note()
                               for _ in range(num_samples)]

        return df

    def _generate_fraudulent_transactions(self, num_samples: int) -> pd.DataFrame:
        """
        Generate fraudulent transactions.

        These have suspicious patterns:
        - Unusually large amounts
        - Strange timing (late night)
        - High-risk countries
        - Suspicious descriptions
        - Multiple rapid transactions
        """
        logger.info("🚨 Generating fraudulent transactions...")

        data = {
            # === TABULAR FEATURES (Numbers and Categories) ===

            # Fraud transactions tend to be larger or very small (testing)
            'transaction_amount': np.concatenate([
                np.random.lognormal(mean=7.0, sigma=1.5, size=int(num_samples*0.7)),  # Large
                np.random.uniform(0.01, 10, size=int(num_samples*0.3))  # Small test amounts
            ])[:num_samples],

            # Fraudsters use newer accounts (compromised or created for fraud)
            'account_age_days': np.random.exponential(scale=180, size=num_samples),

            # Many transactions in short time (trying to maximize damage)
            'num_transactions_24h': np.random.poisson(lam=8, size=num_samples),

            # Average amount might be low (account recently compromised)
            'avg_transaction_amount': np.random.lognormal(mean=3.5, sigma=1.5, size=num_samples),

            # Fraud often happens late at night or early morning
            'transaction_hour': np.random.choice(list(range(0, 6)) + list(range(22, 24)),
                                                size=num_samples),

            # Fraud can happen any day
            'transaction_day_of_week': np.random.choice(range(7), size=num_samples),

            # Stolen accounts could be any age
            'customer_age': np.random.normal(loc=40, scale=20, size=num_samples).astype(int),

            # Credit scores vary for compromised accounts
            'credit_score': np.random.normal(loc=650, scale=100, size=num_samples).astype(int),

            # More failed logins (account takeover attempts)
            'num_failed_logins': np.random.choice([0, 1, 2, 3, 5, 10], size=num_samples,
                                                 p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]),

            # Label: 1 = fraudulent
            'is_fraud': np.ones(num_samples, dtype=int)
        }

        # Clip to realistic ranges
        data['transaction_amount'] = np.clip(data['transaction_amount'], 0.01, 100000)
        data['account_age_days'] = np.clip(data['account_age_days'], 1, 7300)
        data['customer_age'] = np.clip(data['customer_age'], 18, 90)
        data['credit_score'] = np.clip(data['credit_score'], 300, 850)

        df = pd.DataFrame(data)

        # === CATEGORICAL FEATURES ===
        df['source_country'] = [self._get_risky_country() for _ in range(num_samples)]
        df['destination_country'] = [self._get_risky_country() for _ in range(num_samples)]
        df['account_type'] = np.random.choice(['checking', 'savings', 'credit'],
                                             size=num_samples, p=[0.7, 0.2, 0.1])

        # === TEXT FEATURES ===
        df['transaction_description'] = [self._get_suspicious_description()
                                        for _ in range(num_samples)]
        df['merchant_category'] = [self._get_risky_merchant()
                                  for _ in range(num_samples)]
        df['customer_notes'] = [self._get_suspicious_note()
                               for _ in range(num_samples)]

        return df

    # =========================================================================
    # Helper Methods for Generating Realistic Text Data
    # =========================================================================

    def _get_legitimate_country(self) -> str:
        """Return a low-risk country for legitimate transactions."""
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia',
                    'Japan', 'South Korea', 'Singapore', 'Switzerland']
        return random.choice(countries)

    def _get_risky_country(self) -> str:
        """Return countries often associated with fraud (for educational purposes only!)."""
        # Note: This is for educational ML purposes - real fraud can come from anywhere
        countries = ['Nigeria', 'Russia', 'China', 'Romania', 'Ghana', 'Pakistan',
                    'Unknown', 'Anonymous', 'VPN_Masked']
        return random.choice(countries)

    def _get_legitimate_description(self) -> str:
        """Generate realistic transaction descriptions for legitimate transactions."""
        templates = [
            "Purchase at {company}",
            "Online payment to {company}",
            "Bill payment - {company}",
            "Transfer to {person}",
            "ATM withdrawal at {location}",
            "Restaurant payment - {company}",
            "Grocery shopping at {company}",
            "Gas station purchase",
            "Monthly subscription {company}",
            "Salary deposit from {company}"
        ]
        template = random.choice(templates)
        return template.format(
            company=self.fake.company(),
            person=self.fake.name(),
            location=self.fake.city()
        )

    def _get_suspicious_description(self) -> str:
        """Generate suspicious transaction descriptions."""
        templates = [
            "Wire transfer to overseas account",
            "Large cash withdrawal",
            "Transfer to unknown recipient",
            "International wire transfer - urgent",
            "Cryptocurrency purchase - large amount",
            "Money transfer service - {country}",
            "Online casino deposit",
            "High-risk merchant payment",
            "Transfer to shell company",
            "Anonymous payment gateway",
            "Gift card purchase - bulk",
            "Money order - large amount"
        ]
        template = random.choice(templates)
        return template.format(country=self._get_risky_country())

    def _get_legitimate_merchant(self) -> str:
        """Return legitimate merchant categories."""
        categories = [
            'Grocery', 'Gas Station', 'Restaurant', 'Retail', 'Healthcare',
            'Utilities', 'Insurance', 'Education', 'Transportation', 'Entertainment'
        ]
        return random.choice(categories)

    def _get_risky_merchant(self) -> str:
        """Return high-risk merchant categories."""
        categories = [
            'Casino', 'Online Gambling', 'Cryptocurrency', 'Money Transfer',
            'Adult Services', 'Offshore Banking', 'High-Risk Retail',
            'Unregistered Charity', 'Anonymous Service', 'Dark Web'
        ]
        return random.choice(categories)

    def _get_legitimate_note(self) -> str:
        """Generate normal customer notes."""
        notes = [
            "Regular monthly payment",
            "Authorized transaction",
            "Expected purchase",
            "",  # Often no note
            "Planned expense",
            "Budgeted transaction"
        ]
        return random.choice(notes)

    def _get_suspicious_note(self) -> str:
        """Generate suspicious customer notes or flags."""
        notes = [
            "Disputed transaction",
            "Did not authorize",
            "Possible fraud",
            "Account compromised?",
            "Unusual activity",
            "Contact customer urgently",
            ""
        ]
        return random.choice(notes)

    def save_data(self, data: pd.DataFrame, output_path: str):
        """
        Save generated data to CSV file.

        Parameters:
        -----------
        data : pd.DataFrame
            The generated dataset
        output_path : str
            Where to save the CSV file
        """
        logger.info(f"💾 Saving data to {output_path}")
        data.to_csv(output_path, index=False)
        logger.info(f"✅ Data saved successfully!")

        # Print some statistics
        logger.info(f"\n📊 Dataset Statistics:")
        logger.info(f"   Total transactions: {len(data)}")
        logger.info(f"   Fraudulent: {data['is_fraud'].sum()} ({data['is_fraud'].mean()*100:.2f}%)")
        logger.info(f"   Legitimate: {(1-data['is_fraud']).sum()} ({(1-data['is_fraud'].mean())*100:.2f}%)")
        logger.info(f"   Features: {len(data.columns)}")


def main():
    """
    Main function to run the data generation.
    This is what gets executed when you run this script!
    """
    print("\n" + "="*80)
    print("🏦 BANKING TRANSACTION DATA GENERATOR FOR AML FRAUD DETECTION")
    print("="*80 + "\n")

    # Create generator
    generator = BankingDataGenerator(
        num_transactions=100000,
        fraud_ratio=0.02,
        random_seed=42
    )

    # Generate data
    data = generator.generate_data()

    # Save to file
    output_path = './data/raw/banking_transactions.csv'
    generator.save_data(data, output_path)

    # Show sample of the data
    print("\n" + "="*80)
    print("📋 SAMPLE OF GENERATED DATA (First 5 rows):")
    print("="*80)
    print(data.head())

    print("\n" + "="*80)
    print("📊 DATA TYPES:")
    print("="*80)
    print(data.dtypes)

    print("\n✅ Data generation complete! Check './data/raw/banking_transactions.csv'\n")


if __name__ == "__main__":
    main()
